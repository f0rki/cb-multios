#!/usr/bin/env python
import argparse
import glob
import os
import subprocess
import sys
import logging
import datetime
from collections import OrderedDict

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
    log.setLevel(loglevel)
    if console and colorlog is not None:
        handler = colorlog.StreamHandler()
        fmt = '%(log_color)s%(levelname)-8s%(reset)s : %(name)s :: %(message)s'
        fmter = colorlog.ColoredFormatter(fmt)
        handler.setFormatter(fmter)
        log.addHandler(handler)
    elif console:
        fmt = '%(levelname)-8s : %(name)s :: %(message)s'
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(handler)

    if logfile is not None:
        log.debug("logging to file '{}'".format(logfile))
        handler = logging.FileHandler(logfile)
        fmt = '%(asctime)s ; %(levelname)s ; %(name)s ; %(message)s'
        handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M"))
        log.addHandler(handler)

    return log


class Score:
    """Contains the results of a test"""

    def __init__(self):
        self.passed = 0
        self.total = 0

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

    def __init__(self, chal_name):
        self.name = chal_name

        # Directories used in testing
        self.chal_dir = os.path.join(CHAL_DIR, self.name)
        self.bin_dir = os.path.join(self.chal_dir, 'bin')
        self.pov_dir = os.path.join(self.chal_dir, 'pov')
        self.poll_dir = os.path.join(self.chal_dir, 'poller')

        # Keep track of success for each build variant
        self.variants = OrderedDict({"": True,
                                     "patched": False,
                                     "wit": False,
                                     "skipwit": True})
        self.povs = {k: Score() for k in self.variants}
        self.polls = {k: Score() for k in self.variants}

    @property
    def povs_total(self):
        return sum(score.total for score in self.povs.itervalues())

    @property
    def polls_total(self):
        return sum(score.total for score in self.polls.itervalues())

    @property
    def povs_passed(self):
        return sum(score.passed for score in self.povs.itervalues())

    @property
    def polls_passed(self):
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

    @staticmethod
    def parse_results(output):
        """ Parses out the number of passed and failed tests from cb-test output

        Args:
            output (str): Raw output from running cb-test
        Returns:
            (int, int): # of tests run, # of tests passed
        """
        # If the test failed to run, consider it failed
        if 'TOTAL TESTS' not in output:
            log.warning('there was an error running a test """%s"""', output)
            return 0, 0

        if 'timed out' in output:
            log.warning('test(s) timed out')

        # Parse out results
        total = int(output.split('TOTAL TESTS: ')[1].split('\n')[0])
        passed = int(output.split('TOTAL PASSED: ')[1].split('\n')[0])
        return total, passed

    def run_test(self, bin_names, xml_dir, should_core=False):
        """ Runs a test using cb-test and saves the result

        Args:
            bin_names (list of str): Name of the binary being tested
            xml_dir (str): Directory containing all xml tests
            score (Score): Object to store the results in
            should_core (bool): If the binary is expected to crash with these tests
        """
        cb_cmd = ['./cb-test',
                  '--directory', self.bin_dir,
                  '--xml_dir', xml_dir,
                  '--concurrent', '4',
                  '--timeout', '5',
                  '--negotiate_seed', '--cb'] + bin_names
        if should_core:
            cb_cmd += ['--should_core']

        p = subprocess.Popen(cb_cmd, cwd=TEST_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        total, passed = self.parse_results(out)
        return total, passed

    def run_against_dir(self, xml_dir, scores, is_pov=False):
        """ Runs all tests in a given directory
        against the patched and unpatched versions of a binary

        Args:
            xml_dir (str): Directory containing all xml tests
            score (Score): Object to store the results in
            is_pov (bool): If the files in this directory are POVs
        """
        # Check if there are any tests available in this directory
        tests = glob.glob(os.path.join(xml_dir, '*.xml'))
        tests += glob.glob(os.path.join(xml_dir, '*.pov'))
        if len(tests) == 0:
            log.info('None found')
            return

        # *2 because each test is run against the patched and unpatched binary
        log.info('Running {} test(s)'.format(len(tests) * 2))

        # Collect the names of binaries to be tested
        cb_dirs = glob.glob(os.path.join(self.chal_dir, 'cb_*'))
        if len(cb_dirs) > 0:
            # There are multiple binaries in this challenge
            bin_names = ['{}_{}'.format(self.name, i + 1) for i in range(len(cb_dirs))]
        else:
            bin_names = [self.name]

        # Keep track of old pass/totals
        all_total, all_passed = 0, 0

        # Run the tests
        for pf in self.variants:
            log.info("Running tests for variant '%s'", pf)
            score = scores[pf]
            if is_pov:
                should_core = self.variants[pf]
            else:
                should_core = False
            t, p = self.run_test(['{}{}{}'.format(b, "_" if pf else "", pf)
                                  for b in bin_names],
                                 xml_dir, should_core=should_core)
            log.info('%s for variant %s => Passed %d/%d',
                     "POV" if is_pov else "POLL", pf, p, t)
            score.total += t
            score.passed += p
            all_total += t
            all_passed += p

        # Display resulting totals
        log.info('{} {} => Passed {}/{}'.format(self.name,
                                                "POV" if is_pov else "POLL",
                                                all_passed,
                                                all_total))

    def run(self):
        """Runs all tests for this challenge binary"""
        log.info('Testing {}...'.format(self.name))

        # Test POVs
        if Tester.povs_enabled:
            log.info('running POVs')
            self.run_against_dir(self.pov_dir, self.povs, is_pov=True)

        # Test POLLs
        if Tester.polls_enabled:
            for subdir in listdir(self.poll_dir):
                log.info('running POLL {}'.format(subdir))
                self.run_against_dir(os.path.join(self.poll_dir, subdir), self.polls)
        log.info('Done testing {} => Passed {}/{} tests'.format(self.name, self.passed, self.total))


def test_challenges(chal_names):
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
    testers = map(Tester, chals)
    for test in testers:
        test.run()

    return testers


def get_testrun_info():
    """
    Return information on how the current test results where produced

    Returns:
        A dictionary containing 'git-commit', 'buildtime', 'c-compiler',
        'cpp-compiler'

    """
    info = OrderedDict()
    now = datetime.datetime.now()
    info['time'] = now.strftime("%Y-%M-%D %H:%M (UTC %z)")
    info['git-commit'] = subprocess.check_output(["git", "log", "-1",
                                                  "--format=%H"]).strip()
    cmakecache = os.path.join(BUILD_DIR, "CMakeCache.txt")
    with open(cmakecache) as f:
        for line in f.readlines():
            if line.startswith("CMAKE_C_COMPILER:FILEPATH="):
                info["c-compiler"] = line.split("=")[1].strip()
            elif line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                info["cpp-compiler"] = line.split("=")[1].strip()
    return info


def generate_xlsx(path, tests):
    """ Generates an excel spreadsheet containing the results of all tests

    Args:
        path (str): Path to save the spreadsheet
        tests (list of Tester): All completed tests
    """
    log.info('Generating excel spreadsheet...')
    # Fix filename
    if not path.endswith('.xlsx'):
        path += '.xlsx'

    wb = xl.Workbook(path)
    ws = wb.add_worksheet()

    # Some cell formats used in the sheet
    fmt_name = wb.add_format({'font_color': '#00ff00', 'bg_color': 'black',
                              'border': 1, 'border_color': '#005500'})
    fmt_perfect = wb.add_format({'bg_color': '#b6d7a8', 'border': 1, 'border_color': '#cccccc'})
    fmt_bad = wb.add_format({'bg_color': '#ea9999', 'border': 1, 'border_color': '#cccccc'})
    fmt_none = wb.add_format({'bg_color': '#ffe599', 'border': 1, 'border_color': '#cccccc'})
    fmt_default = wb.add_format({'bg_color': 'white', 'border': 1, 'border_color': '#cccccc'})

    # Some common format strings
    subtract = '={}-{}'
    add = '={}+{}'
    percent = '=100*{}/MAX(1, {})'

    # Write headers
    cols = ['CB_NAME',
            'POVs Total', 'POVs Passed', 'POVs Failed', '% POVs Passed', '',
            'POLLs Total', 'POLLs Passed', 'POLLs Failed', '% POLLs Passed', '',
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
    def write_formula(row, col_name, formula, formula_col1, formula_col2, fmt=fmt_default):
        # type: (int, str, str, str, str, xl.format.Format) -> None
        ws.write_formula(row, col_to_idx[col_name],
                         formula.format(xlutil.xl_rowcol_to_cell(row, col_to_idx[formula_col1]),
                                        xlutil.xl_rowcol_to_cell(row, col_to_idx[formula_col2])), fmt)

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

        # NOTE: Leaving all of these to be calculated in excel in case you want to manually edit it later
        # POVs
        fmt = select_fmt(test.povs_total, test.povs_passed)
        ws.write_row(row, col_to_idx['POVs Total'],
                     [test.povs_total, test.povs_passed], fmt)
        write_formula(row, 'POVs Failed', subtract, 'POVs Total', 'POVs Passed', fmt)
        write_formula(row, '% POVs Passed', percent, 'POVs Passed', 'POVs Total', fmt)

        # POLLs
        fmt = select_fmt(test.polls_total, test.polls_passed)
        ws.write_row(row, col_to_idx['POLLs Total'],
                     [test.polls_total, test.polls_passed], fmt)
        write_formula(row, 'POLLs Failed', subtract, 'POLLs Total', 'POLLs Passed', fmt)
        write_formula(row, '% POLLs Passed', percent, 'POLLs Passed', 'POLLs Total', fmt)

        # Totals
        fmt = select_fmt(test.total, test.passed)
        write_formula(row, 'Total Tests', add, 'POVs Total', 'POLLs Total', fmt)
        write_formula(row, 'Total Passed', add, 'POVs Passed', 'POLLs Passed', fmt)
        write_formula(row, 'Total Failed', subtract, 'Total Tests', 'Total Passed', fmt)
        write_formula(row, 'Total % Passed', percent, 'Total Passed', 'Total Tests', fmt)

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
    skip_cols = ['', 'CB_NAME', '% POVs Passed', '% POLLs Passed', 'Total % Passed', 'Notes']

    # Totals at bottom
    row += 1
    ws.write(row, 0, 'TOTAL')
    for col_name in cols:
        if col_name not in skip_cols:
            col = col_to_idx[col_name]
            ws.write_formula(row, col, '=SUM({})'.format(xlutil.xl_range(1, col, len(tests), col)))

    # Calculate total %'s
    write_formula(row, '% POVs Passed', percent, 'POVs Passed', 'POVs Total')
    write_formula(row, '% POLLs Passed', percent, 'POLLs Passed', 'POLLs Total')
    write_formula(row, 'Total % Passed', percent, 'Total Passed', 'Total Tests')

    # These columns are ignored in averages
    skip_cols = ['', 'CB_NAME', 'Notes']

    # Averages at bottom
    row += 1
    ws.write(row, 0, 'AVERAGE')
    for col_name in cols:
        if col_name not in skip_cols:
            col = col_to_idx[col_name]
            ws.write_formula(row, col, '=AVERAGE({})'.format(xlutil.xl_range(1, col, len(tests), col)))

    # write some info on how the binaries were built, etc.
    row += 2
    info = get_testrun_info()
    for k, v in info.iteritems():
        ws.write_row(row, 0, (k, v))
        row += 1

    # Done, save the spreadsheet
    wb.close()
    log.info('Done, saved to {}'.format(path))


def listdir(path):
    # type: (str) -> list
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path), key=lambda s: s.lower())


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
                        help='If provided, an excel spreadsheet will be generated and saved here')

    parser.add_argument('-l', '--logfile',
                        default=None, type=str,
                        help='Log output of this script to this file')

    parser.add_argument('-q', '--quiet',
                        default=False, action='store_true',
                        help='Surpress console log output')

    args = parser.parse_args(sys.argv[1:])

    # Disable other tests depending on args
    if args.povs:
        Tester.polls_enabled = False
    if args.polls:
        Tester.povs_enabled = False

    global log
    log = setup_logging(console=(not args.quiet), logfile=args.logfile)

    if args.all:
        log.info('Running tests against all challenges')
        tests = test_challenges(listdir(CHAL_DIR))
    else:
        log.info('Running tests against {} challenge(s)'.format(len(args.chals)))
        tests = test_challenges(args.chals)

    if args.output:
        generate_xlsx(os.path.abspath(args.output), tests)


if __name__ == '__main__':
    main()
