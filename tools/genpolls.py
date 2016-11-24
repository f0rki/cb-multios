#!/usr/bin/env python3

import os
import sys
import logging
import argparse
# python3 subprocess supports the timeout arg
import subprocess as sp
# python3 has the mp.get_context('spawn'), which we apparently need to
# mix sp and mp:  https://bugs.python.org/issue25829
import multiprocessing as mp
from time import sleep

try:
    import colorlog
except ImportError:
    colorlog = None


def setup_logging(console=True, logfile=None,
                  loglevel=logging.INFO, name="genpolls"):
    log = logging.getLogger(name)
    log.handlers = []  # clear all handlers that might already be there
    log.setLevel(loglevel)
    if console and colorlog is not None:
        handler = colorlog.StreamHandler()
        fmt = '%(log_color)s%(levelname)-8s%(reset)s : %(name)s (%(process)d) :: %(message)s'
        fmter = colorlog.ColoredFormatter(fmt)
        handler.setFormatter(fmter)
        log.addHandler(handler)
    elif console:
        fmt = '%(levelname)-8s : %(name)s (%(process)d) :: %(message)s'
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(handler)

    if logfile is not None:
        log.debug("logging to file '{}'".format(logfile))
        handler = logging.FileHandler(logfile)
        fmt = '%(asctime)s; %(process)d ; %(levelname)s ; %(name)s ; %(message)s'
        handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M"))
        log.addHandler(handler)

    return log


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CHAL_DIR = os.path.join(os.path.dirname(TOOLS_DIR), 'processed-challenges')
GEN_POLLS = os.path.join(TOOLS_DIR, "generate-polls", "generate-polls")
GEN_POLLS_CMD = ['python2', '-B', GEN_POLLS, '--count', '500', '--repeat', '0',
                 '--store_seed']
TIMEOUT = (60 * 10)


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
            log.debug("found 'cdll'/'CDLL' in %s", path)
            return True
    except sp.CalledProcessError as e:
        if e.returncode != 1:
            log.exception("grep failed '{}'".format(cmd))
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
    log.debug("'%s' exists? %s", machinepy, os.path.exists(machinepy))
    if not os.path.exists(machinepy):
        log.debug("not generating poll for %s (no machine.py) ", pollid)
        return (path, True, "no machine.py")

    if is_blacklisted_poller(path):
        log.info("skipping blacklisted poller %s", pollid)
        return (path, False, "blacklisted")

    log.info("generating poll %s", pollid)
    for pollnum in (1000, 500, 100, 10, 5, 3, 1):
        lasttimeouted = False
        cmd = GEN_POLLS_CMD + [os.path.join(path, "machine.py"),
                               os.path.join(path, "state-graph.yaml"),
                               path]
        log.debug("try with --count %d", pollnum)
        cmd[cmd.index("--count") + 1] = str(pollnum)
        spwd = os.path.abspath(os.path.join(path, "..", ".."))
        log.debug("launching process: '%s' in working dir '%s'",
                  " ".join(cmd), spwd)
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE,
                        cwd=spwd)
        log.debug("setting timeout to %d", TIMEOUT)
        try:
            out, err = proc.communicate(timeout=TIMEOUT)
            out = out.decode('utf-8')
            err = err.decode('utf-8')

            if proc.returncode == 0:
                if out or err:
                    log.debug("child exited:\n-- stdout: --\n%s\n-- stderr: --\n%s",
                              out, err)
                else:
                    log.debug("child %d exited without output", proc.pid)
                log.info("got %d polls for %s", pollnum, pollid)

                # exit loop on success
                break
            else:
                if "Assertion" in err:
                    log.warn("retrying %s because child exited with " +
                             "assertion (returncode %d)" +
                             ":\n-- stdout: --\n%s\n-- stderr: --\n%s",
                             pollid, proc.returncode, out, err)
                    continue
                else:
                    break
        except sp.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            out = out.decode('utf-8')
            err = err.decode('utf-8')
            log.warning("child timeouted:\n-- stdout: --\n%s\n-- stderr: --\n%s",
                        out, err)
            lasttimeouted = True
            continue
        except KeyboardInterrupt:
            proc.kill()
            log.error("Interrupted! stopped child and bailing")
            raise

    if lasttimeouted:
        log.warning("timeout generating poller %s:\n-- stdout: --\n%s\n-- stderr: --\n%s",
                    pollid, out, err)
        return (path, False, "timeout")

    if proc.returncode == 0:
        log.info("success generating poller %s", pollid)
        return (path, True, "")
    else:
        log.warning("failure generating poller %s:\n-- stdout: --\n%s\n-- stderr: --\n%s",
                    pollid, out, err)
        return (path, False, "returncode {}".format(proc.returncode))


def generate_polls(path, args):
    pool = mp.get_context("spawn").Pool(processes=args['jobs'],
                                        initializer=init_from_args,
                                        initargs=(args, True))
    polls = list(find_polls(path))
    log.debug("got polls %s", polls)
    try:
        results = list(pool.imap_unordered(gen_poll, polls))
    except KeyboardInterrupt:
        sleep(2)
        log.info("terminating pool")
        pool.terminate()
        log.info("exiting")
        sys.exit(-1)

    succeeded = sum(1 for _, success, _ in results if success)
    failed = sum(1 for _, success, _ in results if not success)
    timeouted = sum(1 for _, success, why in results
                    if not success and "timeout" in why)
    blacklisted = sum(1 for _, success, why in results
                      if not success and "blacklist" in why)
    log.info("poll generation: %d succeeded, %d failed " +
             "(%d timeouted, %d blacklisted, %d others)",
             succeeded, failed, timeouted, blacklisted,
             (failed - timeouted - blacklisted))
    missing = [(os.path.join(*split_all(path)[-3:]), why)
               for path, success, why in results if not success]
    if missing:
        log.warning("failed to generate polls:\n%s",
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
                        help='Timeout for one generate-polls subprocess (default = 10 min)')

    args = vars(parser.parse_args(sys.argv[1:]))

    init_from_args(args)

    generate_polls(CHAL_DIR, args)


if __name__ == '__main__':
    main()
