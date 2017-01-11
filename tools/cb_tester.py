#!/usr/bin/env python
import argparse
import csv
import datetime
import glob
import json
import logging
import os
import signal
import sys
# import subprocess

from collections import OrderedDict
from multiprocessing import cpu_count

import subprocess32 as subprocess

try:
    import cPickle as pickle
except ImportError:
    import pickle

import xlsxwriter as xl  # pip install xlsxwriter
import xlsxwriter.utility as xlutil

try:
    import colorlog
except ImportError:
    colorlog = None


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CHAL_DIR = os.path.join(os.path.dirname(TOOLS_DIR), 'processed-challenges')
TEST_DIR = os.path.join(TOOLS_DIR, 'cb-testing')
BUILD_DIR = os.path.join(os.path.dirname(TOOLS_DIR), 'build')


def setup_logging(console=True, logfile=None, loglevel=logging.INFO,
                  name="cb_tester"):
    log = logging.getLogger(name)
    log.handlers = []  # remove previous handlers
    log.setLevel(loglevel)
    timefmt = "%Y-%m-%d %H:%M"
    if console and colorlog is not None:
        handler = colorlog.StreamHandler()
        fmt = '%(log_color)s%(levelname)-8s%(reset)s : %(asctime)s :  %(message)s'  # NOQA
        fmter = colorlog.ColoredFormatter(fmt, timefmt)
        handler.setFormatter(fmter)
        log.addHandler(handler)
    elif console:
        fmt = '%(levelname)-8s : %(asctime)s : %(message)s'
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, timefmt))
        log.addHandler(handler)

    if logfile is not None:
        log.debug("logging to file '{}'".format(logfile))
        handler = logging.FileHandler(logfile)
        fmt = '%(asctime)s ; %(levelname)s ; %(name)s ; %(message)s'
        handler.setFormatter(logging.Formatter(fmt, timefmt + ":%s"))
        log.addHandler(handler)

    return log


class TesterLogAdapter(logging.LoggerAdapter):
    """Add the name of the current cb to log messages"""

    def process(self, msg, kwargs):
        return "[{}] {}".format(self.extra['cbname'], msg), kwargs


class TestTimeoutExpired(Exception):
    pass


class Score:
    """Contains the results of a test"""

    def __init__(self):
        self.passed = 0
        self.total = 0
        self.timeouted = 0

    @property
    def failed(self):
        """Number of failed tests"""
        return self.passed - self.total


class Tester:
    """Tests and keeps track of the results of a single challenge binary"""

    # These determine which types of tests will be run
    # Both are enabled by default
    povs_enabled = True
    polls_enabled = True

    def __init__(self, chal_name, variants=None,
                 test_timeout=(60 * 60), test_tries=3, cb_timeout=5):
        self.name = chal_name
        self.finished = False

        self.test_tries = test_tries
        self.test_timeout = test_timeout
        self.cb_timeout = cb_timeout

        # Directories used in testing
        self.chal_dir = os.path.join(CHAL_DIR, self.name)
        self.bin_dir = os.path.join(self.chal_dir, 'bin')
        self.pov_dir = os.path.join(self.chal_dir, 'pov')
        self.poll_dir = os.path.join(self.chal_dir, 'poller')

        # Keep track of success for each build variant
        self.variants = OrderedDict({"": True,
                                     "patched": False})
        if variants:
            self.variants.update(variants)

        self.reset()

        # keep track of test runtimes
        self.runtime = {v: {'povs': {}, 'polls': {}} for v in self.variants}

        self._setup_log_adapter()

    def reset(self):
        self.povs = {k: Score() for k in self.variants}
        self.polls = {k: Score() for k in self.variants}
        self.finished = False

    def _setup_log_adapter(self):
        self.log = TesterLogAdapter(log, {'cbname': self.name})

    @property
    def povs_total(self):
        """Total number of povs"""
        return sum(score.total for score in self.povs.itervalues())

    @property
    def polls_total(self):
        """Total number of polls"""
        return sum(score.total for score in self.polls.itervalues())

    @property
    def povs_passed(self):
        """Number of passed povs"""
        return sum(score.passed for score in self.povs.itervalues())

    @property
    def polls_passed(self):
        """Number of passed polls"""
        return sum(score.passed for score in self.polls.itervalues())

    @property
    def passed(self):
        """Number of passed tests"""
        return (sum(score.passed for score in self.povs.itervalues()) +
                sum(score.passed for score in self.polls.itervalues()))

    @property
    def total(self):
        """Total number of tests run"""
        return (sum(score.total for score in self.povs.itervalues()) +
                sum(score.total for score in self.polls.itervalues()))

    @property
    def failed(self):
        """Number of failed tests"""
        return self.total - self.passed

    def parse_results(self, output):
        """
        Parse out the number of passed and failed tests from cb-test output.

        Args:
            output (str): Raw output from running cb-test
        Returns:
            (int, int): # of tests run, # of tests passed
        """
        # If the test failed to run, consider it failed
        if 'TOTAL TESTS' not in output:
            self.log.warning('there was an error running a test """{}"""'
                             .format(output))
            return 0, 0, 0

        if 'timed out' in output:
            timedout = output.count("timed out")
            self.log.warning('{} test(s) timed out'.format(timedout))
        else:
            timedout = 0

        # Parse out results
        total = int(output.split('TOTAL TESTS: ')[1].split('\n')[0])
        passed = int(output.split('TOTAL PASSED: ')[1].split('\n')[0])
        return total, passed, timedout

    def run_test(self, bin_names, xml_dir, should_core=False):
        """
        Run a test using cb-test and saves the result.

        Args:
            bin_names (list of str): Name of the binary being tested
            xml_dir (str): Directory containing all xml tests
            score (Score): Object to store the results in
            should_core (bool): If the binary is expected to crash
        """
        cb_cmd = ['./cb-test',
                  '--directory', self.bin_dir,
                  '--xml_dir', xml_dir,
                  '--concurrent', str(cpu_count()),
                  '--timeout', str(self.cb_timeout),
                  '--negotiate_seed', '--cb'] + bin_names
        if should_core:
            cb_cmd += ['--should_core']

        for i in range(self.test_tries):
            self.log.debug("running command '{}' in dir '{}' (try {} / {})"
                           .format(" ".join(cb_cmd), TEST_DIR,
                                   i + 1, self.test_tries))
            with subprocess.Popen(cb_cmd,
                                  cwd=TEST_DIR,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  preexec_fn=os.setsid) as p:
                try:
                    out, err = p.communicate(timeout=self.test_timeout)
                    break  # out of tries loop on success
                except subprocess.TimeoutExpired:
                    self.log.warning(("timeout after {} sec while running "
                                      "cb-test (try {} / {}) cmd='{}'")
                                     .format(self.test_timeout,
                                             i + 1,
                                             self.test_tries,
                                             " ".join(cb_cmd)))
                    try:
                        # hopefully this will destroy all remaining
                        # subprocesses etc.
                        os.killpg(p.pid, signal.SIGKILL)
                    except:
                        log.debug("got exception during kill: ",
                                  exc_info=sys.exc_info())
        else:
            self.log.warning(("Test timeouted after {} tries for {}"
                              " with command '{}'")
                             .format(self.test_tries, self.name,
                                     " ".join(cb_cmd)))
            raise TestTimeoutExpired()

        total, passed, timedout = self.parse_results(out)
        return total, passed, timedout

    def run_against_dir(self, xml_dir, scores, is_pov=False):
        """
        Run all tests in a given directory against the patched and unpatched
        versions of a binary.

        Args:
            xml_dir (str): Directory containing all xml tests
            score (Score): Object to store the results in
            is_pov (bool): If the files in this directory are POVs
        """
        # Check if there are any tests available in this directory
        tests = glob.glob(os.path.join(xml_dir, '*.xml'))
        tests += glob.glob(os.path.join(xml_dir, '*.pov'))
        if len(tests) == 0:
            self.log.info('No tests found in "{}"'.format(xml_dir))
            return

        self.log.info('Running {} test(s)'
                      .format(len(tests) * len(self.variants)))

        # Collect the names of binaries to be tested
        cb_dirs = glob.glob(os.path.join(self.chal_dir, 'cb_*'))
        if len(cb_dirs) > 0:
            # There are multiple binaries in this challenge
            bin_names = ['{}_{}'.format(self.name, i + 1)
                         for i in range(len(cb_dirs))]
        else:
            bin_names = [self.name]

        # Keep track of old pass/totals
        all_total, all_passed = 0, 0
        all_timeouted = 0

        # Run the tests
        for pf in self.variants:
            self.log.info("Running tests for variant '{}'".format(pf))
            score = scores[pf]
            if is_pov:
                should_core = self.variants[pf]
            else:
                should_core = False

            start = datetime.datetime.now()
            try:
                t, p, to = self.run_test(['{}{}{}'
                                          .format(b, "_" if pf else "", pf)
                                          for b in bin_names],
                                         xml_dir, should_core=should_core)

            except TestTimeoutExpired:
                t, p, to = len(tests), 0, len(tests)

            stop = datetime.datetime.now()
            dur = stop - start
            self.runtime[pf]['povs' if is_pov else 'polls'][xml_dir] = dur
            mins, secs = divmod(dur.total_seconds(), 60)

            self.log.info(('{} for variant "{}" => Passed {}/{} '
                           '(with {} timeouts, took {} min {} sec)')
                          .format("POV" if is_pov else "POLL", pf, p, t, to,
                                  mins, secs))
            score.total += t
            score.passed += p

            # TODO: remove this ugly hack
            if "timeouted" not in score.__dict__:
                score.timeouted = 0
            score.timeouted += to
            all_total += t
            all_passed += p
            all_timeouted += to

        # Display resulting totals
        self.log.info('{} {} => Passed {}/{}'
                      .format(self.name, "POV" if is_pov else "POLL",
                              all_passed, all_total))

    def run(self):
        """Run all tests for this challenge binary"""
        self.log.info('Testing {}...'.format(self.name))

        # Test POVs
        if Tester.povs_enabled:
            self.log.info('running POVs')
            self.run_against_dir(self.pov_dir, self.povs, is_pov=True)

        # Test POLLs
        if Tester.polls_enabled:
            for subdir in listdir(self.poll_dir):
                self.log.info('running POLL {}'.format(subdir))
                self.run_against_dir(os.path.join(
                    self.poll_dir, subdir), self.polls)
        self.log.info('Done testing {} => Passed {}/{} tests'
                      .format(self.name, self.passed, self.total))
        self.finished = True


def test_challenges(chal_names, variants, previous_testers,
                    test_tries, test_timeout, cb_timeout):
    # type: (list) -> list
    # Filter out any challenges that don't exist
    chals = []
    for c in chal_names:
        cdir = os.path.join(CHAL_DIR, c)
        if not os.path.isdir(cdir):
            log.warning('Challenge "{}" does not exist, skipping'.format(c))
            continue

        # Skip duplicates
        if c in chals:
            log.debug('Ignoring duplicate "{}"'.format(c))
            continue

        chals.append(c)

    # Create and run all testers
    previous_testers = {tester.name: tester for tester in previous_testers}
    testers = {tester.name: tester
               for tester in map(lambda y: Tester(y,
                                                  variants=variants,
                                                  test_tries=test_tries,
                                                  test_timeout=test_timeout,
                                                  cb_timeout=cb_timeout),
                                 chals)
               if tester.name not in previous_testers}
    testers.update(previous_testers)

    # this can be a pretty long running loop. So we make it interruptible,
    # without loosing data about done tests...
    try:
        for i, test in enumerate(testers.values()):
            if test.finished:
                log.info("Skipping previously finished test {}/{} '{}'"
                         .format(i + 1, len(testers), test.name))
            else:
                log.info("Running test {}/{} '{}'"
                         .format(i + 1, len(testers), test.name))
                test.run()
    except KeyboardInterrupt:
        log.info("User abort during test {}/{} '{}'"
                 .format(i + 1, len(testers), test.name))
        # clear results of last test
        test.reset()
    except:
        log.warning("Received exception during test {}/{} '{}'"
                    .format(i + 1, len(testers), test.name),
                    exc_info=sys.exc_info())
        test.reset()

    return list(testers.values())


def get_testrun_info():
    """
    Return information on how the current test results where produced.
    Currently this contains:

      - 'git-commit'
      - 'buildtime'
      - 'c-compiler'
      - 'c-compiler-version'
      - 'cpp-compiler'
      - 'cpp-compiler-version'

    Returns:
        A dictionary containing test environment information.

    """
    info = OrderedDict()
    now = datetime.datetime.now()
    info['finish-time'] = now.strftime("%Y-%M-%D %H:%M (UTC %z)")
    info['git-commit'] = subprocess.check_output(["git", "log", "-1",
                                                  "--format=%H"]).strip()
    cmakecache = os.path.join(BUILD_DIR, "CMakeCache.txt")
    with open(cmakecache) as f:
        for line in f.readlines():
            if line.startswith("CMAKE_C_COMPILER:FILEPATH="):
                info["c-compiler"] = line.split("=")[1].strip()
            elif line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                info["cpp-compiler"] = line.split("=")[1].strip()

    for k in ('c-compiler', 'cpp-compiler'):
        if k in info:
            try:
                o = subprocess.check_output([info[k], '--version'])
                o = o.split("\n")[0]
                info[k + '-version'] = o.strip()
            except subprocess.CalledProcessError:
                log.warning("compiler '{}' doesn't support '--version'"
                            .format(info[k]))
    return info


def save_tests(tests, state_file):
    log.info("saving test state to {}".format(state_file))
    with open(state_file, "wb") as f:
        # do not pickle unpicklable LogAdapter wrapper
        for test in tests:
            test.log = None
        pickle.dump(tests, f)


def load_tests(state_file):
    if not os.path.exists(state_file):
        log.warning("Previous state_file '{}' does not exist"
                    .format(state_file))
        return []
    log.info("loading test state from {}".format(state_file))
    tests = []
    with open(state_file, "rb") as f:
        tests = pickle.load(f)
    for test in tests:
        test._setup_log_adapter()
    return tests


def generate_xlsx(path, tests):
    """ Generates an excel spreadsheet containing the results of all tests

    Args:
        path (str): Path to save the spreadsheet
        tests (list of Tester): All completed tests
    """
    if not tests:
        log.error("No finished testcases!")

    log.info('Generating excel spreadsheet...')
    # Fix filename
    if not path.endswith('.xlsx'):
        path += '.xlsx'

    wb = xl.Workbook(path)
    ws = wb.add_worksheet()

    # Some cell formats used in the sheet
    fmt_name = wb.add_format({'font_color': '#00ff00', 'bg_color': 'black',
                              'border': 1, 'border_color': '#005500'})
    fmt_perfect = wb.add_format(
        {'bg_color': '#b6d7a8', 'border': 1, 'border_color': '#cccccc'})
    fmt_bad = wb.add_format(
        {'bg_color': '#ea9999', 'border': 1, 'border_color': '#cccccc'})
    fmt_none = wb.add_format(
        {'bg_color': '#ffe599', 'border': 1, 'border_color': '#cccccc'})
    fmt_default = wb.add_format(
        {'bg_color': 'white', 'border': 1, 'border_color': '#cccccc'})

    # Some common format strings
    subtract = '={}-{}'
    add = '={}+{}'
    percent = '=100*{}/MAX(1, {})'

    # Write headers
    cols = ['CB_NAME',
            'POVs Total', 'POVs Passed', 'POVs Failed', '% POVs Passed', '',
            'POLLs Total', 'POLLs Passed', 'POLLs Failed', '% POLLs Passed',
            '',
            'Total Tests', 'Total Passed', 'Total Failed', 'Total % Passed',
            'Notes', '']
    for pf in tests[0].variants:
        for x in ('POVs Total', 'POVs Passed'):
            cols.append("{} {}".format(pf.replace("_", ""), x))
        for x in ('POLLs Total', 'POLLs Passed'):
            cols.append("{} {}".format(pf.replace("_", ""), x))
        cols.append('')
    row = 0
    ws.write_row(row, 0, cols)

    # Helper map for getting column indices
    col_to_idx = {val: i for i, val in enumerate(cols)}

    # Helper for writing formulas that use two cells
    def write_formula(row, col_name, formula, formula_col1, formula_col2,
                      fmt=fmt_default):
        # type: (int, str, str, str, str, xl.format.Format) -> None
        fcol1 = xlutil.xl_rowcol_to_cell(row, col_to_idx[formula_col1])
        fcol2 = xlutil.xl_rowcol_to_cell(row, col_to_idx[formula_col2])
        ws.write_formula(row, col_to_idx[col_name],
                         formula.format(fcol1, fcol2), fmt)

    # Helper for choosing the right format for a cell
    def select_fmt(total, passed):
        # type: (int, int) -> xl.format.Format
        if total == 0:
            return fmt_none
        elif total == passed:
            return fmt_perfect
        elif passed == 0:
            return fmt_bad
        return fmt_default

    # Add all test data
    for test in tests:
        row += 1

        # Write the challenge name
        ws.write(row, 0, test.name, fmt_name)

        # NOTE: Leaving all of these to be calculated in excel in case you want
        # to manually edit it later
        # POVs
        fmt = select_fmt(test.povs_total, test.povs_passed)
        ws.write_row(row, col_to_idx['POVs Total'],
                     [test.povs_total, test.povs_passed], fmt)
        write_formula(row, 'POVs Failed', subtract,
                      'POVs Total', 'POVs Passed', fmt)
        write_formula(row, '% POVs Passed', percent,
                      'POVs Passed', 'POVs Total', fmt)

        # POLLs
        fmt = select_fmt(test.polls_total, test.polls_passed)
        ws.write_row(row, col_to_idx['POLLs Total'],
                     [test.polls_total, test.polls_passed], fmt)
        write_formula(row, 'POLLs Failed', subtract,
                      'POLLs Total', 'POLLs Passed', fmt)
        write_formula(row, '% POLLs Passed', percent,
                      'POLLs Passed', 'POLLs Total', fmt)

        # Totals
        fmt = select_fmt(test.total, test.passed)
        write_formula(row, 'Total Tests', add,
                      'POVs Total', 'POLLs Total', fmt)
        write_formula(row, 'Total Passed', add,
                      'POVs Passed', 'POLLs Passed', fmt)
        write_formula(row, 'Total Failed', subtract,
                      'Total Tests', 'Total Passed', fmt)
        write_formula(row, 'Total % Passed', percent,
                      'Total Passed', 'Total Tests', fmt)

        lastcol = col_to_idx['Notes']
        curcol = lastcol + 2

        for postf in test.variants:
            povs, polls = test.povs[postf], test.polls[postf]
            # POVs
            fmt = select_fmt(povs.total, povs.passed)
            rowdata = [povs.total, povs.passed]
            ws.write_row(row, curcol, rowdata, fmt)
            curcol += len(rowdata)

            # POLLs
            fmt = select_fmt(polls.total, polls.passed)
            rowdata = [polls.total, polls.passed]
            ws.write_row(row, curcol, rowdata, fmt)
            curcol += len(rowdata) + 1

    # These columns are ignored in totals
    skip_cols = ['', 'CB_NAME', '% POVs Passed',
                 '% POLLs Passed', 'Total % Passed', 'Notes']

    # Totals at bottom
    row += 1
    ws.write(row, 0, 'TOTAL')
    for col_name in cols:
        if col_name not in skip_cols:
            col = col_to_idx[col_name]
            ws.write_formula(row, col, '=SUM({})'.format(
                xlutil.xl_range(1, col, len(tests), col)))

    # Calculate total %'s
    write_formula(row, '% POVs Passed', percent, 'POVs Passed', 'POVs Total')
    write_formula(row, '% POLLs Passed', percent,
                  'POLLs Passed', 'POLLs Total')
    write_formula(row, 'Total % Passed', percent,
                  'Total Passed', 'Total Tests')

    # These columns are ignored in averages
    skip_cols = ['', 'CB_NAME', 'Notes']

    # Averages at bottom
    row += 1
    ws.write(row, 0, 'AVERAGE')
    for col_name in cols:
        if col_name not in skip_cols:
            col = col_to_idx[col_name]
            ws.write_formula(row, col, '=AVERAGE({})'.format(
                xlutil.xl_range(1, col, len(tests), col)))

    # write some info on how the binaries were built, etc.
    row += 2
    info = get_testrun_info()
    for k, v in info.iteritems():
        ws.write_row(row, 0, (k, v))
        row += 1

    # Done, save the spreadsheet
    wb.close()
    log.info('Done, saved to {}'.format(path))


def generate_json(path, tests, pretty_print=False):
    if not tests:
        log.error("No finished testcases!")

    # Fix filename
    if not path.endswith('.json'):
        path += '.json'

    log.info('Generating json results file "{}"'.format(path))

    results = []

    for test in tests:
        log.debug("Got testcase {}".format(test.name))
        o = {'povs': {}, 'polls': {}}
        o['name'] = test.name
        o['povs']['total'] = test.povs_total
        o['povs']['passed'] = test.povs_passed
        o['povs']['failed'] = test.povs_total - test.povs_passed
        o['polls']['total'] = test.polls_total
        o['polls']['passed'] = test.polls_passed
        o['polls']['failed'] = test.polls_total - test.polls_passed

        o['variants'] = {}

        for var in test.variants:
            x = {'povs': {}, 'polls': {}}
            povs, polls = test.povs[var], test.polls[var]
            x['povs']['total'] = povs.total
            x['povs']['passed'] = povs.passed
            x['povs']['failed'] = povs.total - povs.passed

            x['polls']['total'] = polls.total
            x['polls']['passed'] = polls.passed
            x['polls']['failed'] = polls.total - polls.passed

            o['variants'][var] = x

        results.append(o)

    info = get_testrun_info()

    results = {"info": info, "tests": results}

    with open(path, "w") as f:
        if pretty_print:
            json.dump(results, f,
                      separators=(',', ': '),
                      indent=2, sort_keys=True)
        else:
            json.dump(results, f)
        f.write("\n")


def generate_csv(path, tests):
    if not tests:
        log.error("No finished testcases!")

    # Fix filename
    if not path.endswith('.csv'):
        path += '.csv'

    log.info('Generating csv results file "{}"'.format(path))

    with open(path, "wb") as csvfile:
        cw = csv.writer(csvfile)

        titlerow = ["cb_name"]
        for var in tests[0].variants:
            if not var:
                var = "vuln"
            titlerow += [s.format(var) for s in ("{}_polls_total",
                                                 "{}_polls_passed",
                                                 "{}_povs_total",
                                                 "{}_povs_passed")]
        cw.writerow(titlerow)

        for test in tests:
            row = [test.name]
            for var in test.variants:
                povs, polls = test.povs[var], test.polls[var]
                row += [polls.total, polls.passed,
                        povs.total, povs.passed]

            cw.writerow(row)

    info = get_testrun_info()

    with open(path + ".info", "w") as f:
        for k, v in info.iteritems():
            f.write("{} = {}\n".format(k, v))


def listdir(path, hidden=False):
    # type: (str) -> list
    if not os.path.isdir(path):
        return []
    return sorted((p for p in os.listdir(path) if hidden or p[0] != "."),
                  key=lambda s: s.lower())


def main():
    parser = argparse.ArgumentParser()

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('-a', '--all', action='store_true',
                   help='Run tests against all challenge binaries')

    g.add_argument('-c', '--chals', nargs='+', type=str,
                   help='List of challenge names to test')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--povs', action='store_true',
                   help='Only run tests against POVs')

    g.add_argument('--polls', action='store_true',
                   help='Only run tests against POLLS')

    parser.add_argument('-o', '--output',
                        default=None, type=str,
                        help=('If provided, the test results will be saved'
                              'here in the specified output format'
                              '(default = .xlsx)'))

    parser.add_argument('--xlsx',
                        action='store_true',
                        help='Save results as xlsx spreadsheet')

    parser.add_argument('--json',
                        action='store_true',
                        help='Save results as json file')

    parser.add_argument('--csv',
                        action='store_true',
                        help='Save results as csv file')

    parser.add_argument('-l', '--logfile',
                        default=None, type=str,
                        help='Log output of this script to this file')

    parser.add_argument('-q', '--quiet',
                        default=False, action='store_true',
                        help='Surpress console log output')

    parser.add_argument('-V', '--verbose',
                        default=False, action='store_true',
                        help='Logging with debug output')

    parser.add_argument('-v', '--variants',
                        nargs="+",
                        help="the variants that should be tested, format "
                             "'variant_name:vulnerable' e.g. 'patched:False'")

    parser.add_argument('--save-state',
                        action='store_true',
                        help="store the state of unfinished/finished test"
                             "results")

    parser.add_argument('--load-state',
                        action='store_true',
                        help="load the state of cb test results and continue")

    parser.add_argument("--state-file",
                        default=os.path.join(CHAL_DIR, ".state.pickle"),
                        type=str,
                        help="path to the state file")

    parser.add_argument("--test-timeout",
                        default=(60 * 60), type=int,
                        help="how long a test of a CB in total may take until"
                             " it is aborted.")

    parser.add_argument("--test-tries",
                        default=3, type=int,
                        help="how often to try a test in case of timeout")

    parser.add_argument("--cb-timeout",
                        default=5, type=int,
                        help="timeout for a single POV/POLL passed to cb-test")

    args = parser.parse_args(sys.argv[1:])

    # Disable other tests depending on args
    if args.povs:
        Tester.polls_enabled = False
    if args.polls:
        Tester.povs_enabled = False

    global log
    log = setup_logging(console=(not args.quiet),
                        logfile=args.logfile,
                        loglevel=(logging.DEBUG
                                  if args.verbose
                                  else logging.INFO))

    variants = None
    if args.variants:
        variants = OrderedDict()
        for v in args.variants:
            if ":" not in v:
                raise ValueError("Invalid format for variant '{}'".format(v))
            name, vuln = v.split(":")
            if vuln.startswith("F") or vuln.startswith("f") or vuln == "0":
                vuln = False
            elif vuln.startswith("T") or vuln.startswith("t") or vuln == "1":
                vuln = True
            else:
                raise ValueError("Invalid boolean for variant specifier '{}'"
                                 .format(v))
            variants[name] = vuln

    if args.all:
        log.info('Running tests against all challenges')
        chals = listdir(CHAL_DIR)
    else:
        log.info('Running tests against {} challenge(s)'
                 .format(len(args.chals)))
        chals = args.chals

    if args.load_state:
        previous_tests = load_tests(args.state_file)
        log.info("loaded {} tests from previous state"
                 .format(len(previous_tests)))
        log.debug("previous tests: " +
                  ", ".join("{}: {}".format(t.name,
                                            "done" if t.finished
                                            else "pending")
                            for t in previous_tests))
        previous_tests = [test for test in previous_tests
                          if test.finished]
        log.info("Using {} finished tests from previous run"
                 .format(len(previous_tests)))
    else:
        previous_tests = []

    tests = test_challenges(chals, variants, previous_tests,
                            args.test_tries, args.test_timeout,
                            args.cb_timeout)

    if args.save_state:
        save_tests(tests, args.state_file)

    # save only finished tests to results output
    tests = [test for test in tests if test.finished]

    log.info("Finished {} tests".format(len(tests)))

    if args.output:
        if not tests:
            log.error("No finished testcases. Cannot produce output!")
            return

        # default format is xlsx
        if not (args.csv or args.json):
            args.xlsx = True

        if args.xlsx:
            generate_xlsx(os.path.abspath(args.output), tests)
        if args.json:
            generate_json(os.path.abspath(args.output), tests)
        if args.csv:
            generate_csv(os.path.abspath(args.output), tests)


if __name__ == '__main__':
    main()
