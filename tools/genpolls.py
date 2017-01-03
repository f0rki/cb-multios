#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import hashlib
# python3 subprocess supports the timeout arg
import subprocess as sp
# python3 has the mp.get_context('spawn'), which we apparently need to
# mix sp and mp:  https://bugs.python.org/issue25829
import multiprocessing as mp
from time import sleep
from inspect import getargspec

try:
    import colorlog
except ImportError:
    colorlog = None


# FIXME: logging to a file from multiple processes is not such a good idea.
# things might break. There are some codesnippets around on SO, but they are
# all a little cumbersome. https://github.com/jruere/multiprocessing-logging
# promises to do what we want, but it doesn't seem to work with the spawn model
# of creating mp workers.
# Fortunately this doesn't seem to come up on my machines... But maybe it
# should be fixed?
# TODO: maybe we can use threads instead of mp


class BraceMessage(object):

    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fmt).format(*self.args, **self.kwargs)


class BraceStyleAdapter(logging.LoggerAdapter):
    """
    Wraps a logging.Logger object to support the {} style formating syntax
    in log messages.
    """

    def __init__(self, logger):
        self.logger = logger

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, log_kwargs = self.process(msg, kwargs)
            self.logger._log(level, BraceMessage(msg, args, kwargs), (),
                             **log_kwargs)

    def process(self, msg, kwargs):
        return msg, {key: kwargs[key]
                     for key in getargspec(self.logger._log).args[1:]
                     if key in kwargs}


class MultilineFilter(logging.Filter):
    """replace newlines so that it's clear when a log messages starts/stops"""

    def sanitize(self, s):
        return s.replace("\n", "\n | ")

    def filter(self, record):
        if isinstance(record.msg, BraceMessage):
            if "\n" in record.msg.fmt:
                record.msg.fmt = self.sanitize(record.msg.fmt)
            x = []
            for a in record.msg.args:
                if isinstance(a, str) and "\n" in a:
                    a = self.sanitize(a)
                x.append(a)
            record.msg.args = x

            for k, v in record.msg.kwargs:
                if isinstance(v, str) and "\n" in v:
                    record.msg.kwargs[k] = sanitize(v)
        elif isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        return super(MultilineFilter, self).filter(record)


def setup_logging(console=True, logfile=None,
                  loglevel=logging.INFO, name="genpolls"):
    log = logging.getLogger(name)
    log.handlers = []  # clear all handlers that might already be there
    log.setLevel(loglevel)
    timefmt = "%Y-%m-%d %H:%M"
    if console and colorlog is not None:
        handler = colorlog.StreamHandler()
        fmt = '{log_color}{levelname:8s}{reset} : {name} ({process}) : {asctime} :: {message}'
        fmter = colorlog.ColoredFormatter(fmt, timefmt, style='{')
        handler.setFormatter(fmter)
        log.addHandler(handler)
    elif console:
        fmt = '{levelname:8s} : {name} ({process}) : {asctime} :: {message}'
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, timefmt, style='{'))
        log.addHandler(handler)

    if logfile is not None:
        log.debug("logging to file '{}'".format(logfile))
        handler = logging.FileHandler(logfile)
        fmt = '{asctime} ; {levelname} ; {name} ; {process} ; {message}'
        handler.setFormatter(logging.Formatter(fmt, timefmt, style='{'))
        log.addHandler(handler)

    log.addFilter(MultilineFilter())
    _log = BraceStyleAdapter(log)

    return _log


SEED = hashlib.sha512(str("3.141592653589793").encode("ascii") +
                      str("2.718281828459045").encode("ascii")).hexdigest()
TIMEOUT = (60 * 10)
MAX_COUNT = 200
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CHAL_DIR = os.path.join(os.path.dirname(TOOLS_DIR), 'processed-challenges')
GEN_POLLS = os.path.join(TOOLS_DIR, "generate-polls", "generate-polls")
GEN_POLLS_CMD = ['python2', '-B', GEN_POLLS,
                 '--count', '200',
                 '--rounds', '3',
                 '--seed', SEED,
                 '--store_seed']
POLLNUMS = [MAX_COUNT]
i = MAX_COUNT
while i > 1:
    i = int(i // 2)
    POLLNUMS.append(i)


def split_all(path):
    # print(path)
    x = []
    first, second = os.path.split(path)
    x.append(second)
    while first and second:
        first, second = os.path.split(first)
        if second:
            x.append(second)
        else:
            break
    x.reverse()
    return x


def is_blacklisted_poller(path):
    machinepy = os.path.join(path, 'machine.py')
    cmd = " ".join(('grep', '-i', 'cdll', '"' + machinepy + '"', ">/dev/null"))
    try:
        if sp.check_call(cmd, shell=True) == 0:
            log.debug("found 'cdll'/'CDLL' in {}", path)
            return True
    except sp.CalledProcessError as e:
        if e.returncode != 1:
            log.exception("grep failed '{}'", cmd)
    return False


def find_polls(path):
    for chal in os.listdir(path):
        for poller in os.listdir(os.path.join(path, chal, "poller")):
            yield os.path.join(path, chal, "poller", poller)


def init_from_args(args, is_worker=False):
    global log
    log = setup_logging(console=(not args['quiet']),
                        logfile=args['logfile'],
                        loglevel=(logging.DEBUG if args['verbose']
                                  else logging.INFO),
                        name=("genpolls-worker" if is_worker else
                              "genpolls-main"))

    global TIMEOUT
    TIMEOUT = args['timeout']


def gen_poll(path):
    pollid = os.path.join(*split_all(path)[-3:])

    machinepy = os.path.join(path, 'machine.py')
    log.debug("'{}' exists? {}", machinepy, os.path.exists(machinepy))
    if not os.path.exists(machinepy):
        log.debug("not generating poll for {} (no machine.py) ", pollid)
        return (path, True, "no machine.py")

    if is_blacklisted_poller(path):
        log.info("skipping blacklisted poller {}", pollid)
        return (path, False, "blacklisted")

    log.info("generating poll {}", pollid)
    for pollnum in POLLNUMS:
        lasttimeouted = False
        cmd = GEN_POLLS_CMD + [os.path.join(path, "machine.py"),
                               os.path.join(path, "state-graph.yaml"),
                               path]
        log.debug("try with --count {:d}", pollnum)
        cmd[cmd.index("--count") + 1] = str(pollnum)
        spwd = os.path.abspath(os.path.join(path, "..", ".."))
        log.debug("launching process: '{}' in working dir '{}'",
                  " ".join(cmd), spwd)
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE,
                        cwd=spwd)
        log.debug("setting timeout to {:d}", TIMEOUT)
        try:
            out, err = proc.communicate(timeout=TIMEOUT)
            out = out.decode('utf-8')
            err = err.decode('utf-8')

            if proc.returncode == 0:
                if out or err:
                    log.debug("child exited:\n-- stdout: --\n{}\n-- stderr: --\n{}",
                              out, err)
                else:
                    log.debug("child {:d} exited without output", proc.pid)
                log.info("got {:d} polls for {}", pollnum, pollid)

                # exit loop on success
                break
            else:
                if "Assertion" in err:
                    log.warn("retrying {} because child exited with " +
                             "assertion (returncode {:d})" +
                             ":\n-- stdout: --\n{}\n-- stderr: --\n{}",
                             pollid, proc.returncode, out, err)
                    continue
                else:
                    break
        except sp.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            out = out.decode('utf-8')
            err = err.decode('utf-8')
            log.warning("child timeouted:\n-- stdout: --\n{}\n-- stderr: --\n{}",
                        out, err)
            lasttimeouted = True
            continue
        except KeyboardInterrupt:
            proc.kill()
            log.error("Interrupted! stopped child and bailing")
            raise

    if lasttimeouted:
        log.warning("timeout generating poller {}:\n-- stdout: --\n{}\n-- stderr: --\n{}",
                    pollid, out, err)
        return (path, False, "timeout")

    if proc.returncode == 0:
        log.info("success generating poller {}", pollid)
        return (path, True, "")
    else:
        log.warning("failure generating poller {}:\n-- stdout: --\n{}\n-- stderr: --\n{}",
                    pollid, out, err)
        return (path, False, "returncode {}".format(proc.returncode))


def generate_polls(path, args):
    save_list = args['save_list']
    del args['save_list']
    pool = mp.get_context("spawn").Pool(processes=args['jobs'],
                                        initializer=init_from_args,
                                        initargs=(args, True))
    polls = list(find_polls(path))
    log.info("processing {:d} polls", len(polls))
    if log.isEnabledFor(logging.DEBUG):
        pollgen = ((poll.replace(path, "") if poll.startswith(path) else poll)
                   for poll in polls)
        log.debug("processing polls:\n" + "\n".join(pollgen))

    def list_files(path):
        file_set = set([])
        for root, dirs, files in os.walk(path):
            for f in files:
                file_set.add(os.path.join(root, f))
        return file_set

    files_prev = list_files(CHAL_DIR)

    results = None
    remove_files = False
    try:
        results = list(pool.imap_unordered(gen_poll, polls))
    except KeyboardInterrupt:
        log.warning("terminating pool this might take a while!")
        sleep(2)
        pool.terminate()
        remove_files = True

    files_after = list_files(CHAL_DIR)
    generated_files = files_after - files_prev
    if log.isEnabledFor(logging.DEBUG):
        log.debug("generated the following files:\n{}",
                  "\n".join(generated_files))

    if remove_files:
        log.warning("cleaning up generated files")
        for f in generated_files:
            log.debug("removing file {}", f)
            os.remove(f)
    elif save_list:
        save_list.write("\n".join(x.replace(CHAL_DIR, "")
                                  for x in sorted(generated_files)))
        save_list.write("\n")

    if results:
        succeeded = sum(1 for _, success, _ in results if success)
        failed = sum(1 for _, success, _ in results if not success)
        timeouted = sum(1 for _, success, why in results
                        if not success and "timeout" in why)
        blacklisted = sum(1 for _, success, why in results
                          if not success and "blacklist" in why)
        log.info("poll generation: {:d} succeeded, {:d} failed " +
                 "({:d} timeouted, {:d} blacklisted, {:d} others)",
                 succeeded, failed, timeouted, blacklisted,
                 (failed - timeouted - blacklisted))
        missing = [(os.path.join(*split_all(path)[-3:]), why)
                   for path, success, why in results if not success]
        if missing:
            log.warning("failed to generate polls:\n{}",
                        "\n".join("{} -- {}".format(path, why)
                                  for path, why in missing))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--logfile',
                        default=None, type=str,
                        help='Log output of this script to this file')
    parser.add_argument('-q', '--quiet',
                        default=False, action='store_true',
                        help='Surpress console log output')
    parser.add_argument('-v', '--verbose',
                        default=False, action='store_true',
                        help='verbose logging (DEBUG loglevel)')
    parser.add_argument('-j', '--jobs',
                        default=None, type=int,
                        help='Number of parallel jobs (default = cpu_count())')
    parser.add_argument('-t', '--timeout',
                        default=(60 * 10), type=int,
                        help='Timeout for one generate-polls subprocess.')
    parser.add_argument('-S', '--save-list',
                        default=os.path.join(CHAL_DIR, "..", "generated_polls.txt"),
                        type=argparse.FileType('w'),
                        help='save a list of generated files')

    args = vars(parser.parse_args(sys.argv[1:]))

    init_from_args(args)

    generate_polls(CHAL_DIR, args)


if __name__ == '__main__':
    main()
