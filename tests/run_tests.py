"""Cross-backend correctness suite.

    cd tests && python run_tests.py

Runs every registered test case, comparing the `cpp` backend against the
`torch` reference for both dtypes on both devices, forward and backward.
Failures are printed as they happen, summarised at the end, and reported
through the process exit code (0 = everything passed, 1 = something failed),
so this is usable as a CI gate.
"""

import sys
import traceback

import torch

from test_utils import Failure, TestReport, paint
from quat_tests import __tests__ as quat_tests
from se3_tests import __tests__ as se3_tests
from velocities_test import __tests__ as velocity_tests
from interpolation_test import __tests__ as interpolation_tests


SEPARATOR = '=' * 78


def print_summary(reports: list, failures: list) -> None:
    checks_run = sum(r.checks_run for r in reports)
    checks_passed = sum(r.checks_passed for r in reports)
    checks_skipped = sum(r.checks_skipped for r in reports)
    cases_failed = sum(1 for r in reports if r.checks_failed)
    known_failures = [f for r in reports for f in r.known_failures]

    print('')
    print(SEPARATOR)
    print('TEST SUMMARY')
    print(SEPARATOR)
    print('test cases      : {} ({} with failures)'.format(len(reports), cases_failed))
    print('checks run      : {}  (forward + backward, per dtype, per device)'.format(checks_run))
    print(paint('checks passed   : {}'.format(checks_passed),
                'green' if checks_passed else 'yellow'))
    print(paint('checks failed   : {}'.format(len(failures)),
                'red' if failures else 'green'))
    if known_failures:
        print(paint('known failures  : {}  (declared XFAIL, not counted as failures)'.format(
            len(known_failures)), 'yellow'))
    if checks_skipped:
        print(paint('checks skipped  : {}'.format(checks_skipped), 'yellow'))

    if known_failures:
        print('')
        print(paint('KNOWN FAILURES ({}):'.format(len(known_failures)), 'yellow'))
        for failure in known_failures:
            print(paint('  - {} -- {}'.format(failure.header, failure.expected_reason),
                        'yellow'))

    if failures:
        print('')
        print(paint('FAILURES ({}):'.format(len(failures)), 'red', attrs=['bold']))
        for failure in failures:
            print(paint('  - {} {}'.format(failure.header, failure.summary), 'red'))
        print('')
        print(paint('RESULT: FAILED', 'red', attrs=['bold']))
    else:
        print('')
        print(paint('RESULT: PASSED', 'green', attrs=['bold']))
    print(SEPARATOR)


def main() -> int:
    torch.manual_seed(42)
    tests = quat_tests + se3_tests + velocity_tests + interpolation_tests

    reports = []
    failures = []

    for i, test in enumerate(tests):
        name = getattr(test, '__name__', str(test))
        print('[{}/{}] Running {}...'.format(i + 1, len(tests), name))

        try:
            test_obj = test()
        except Exception as exc:                      # noqa: BLE001 - deliberate
            failure = Failure(
                test=name, phase='setup', dtype='-', device='-',
                summary='constructing the test raised {}: {}'.format(
                    type(exc).__name__, exc),
                tb=traceback.format_exc())
            print(paint(failure.render(), 'red'), flush=True)
            report = TestReport(name, checks_run=1, failures=[failure])
            reports.append(report)
            failures.append(failure)
            continue

        try:
            report = test_obj.run()
        except Exception as exc:                      # noqa: BLE001 - deliberate
            # `TestCase.run` already traps per-phase exceptions; anything that
            # escapes it is a harness-level problem.  Report it and carry on
            # with the remaining tests rather than aborting the whole suite.
            failure = Failure(
                test=name, phase='run', dtype='-', device='-',
                summary='test runner raised {}: {}'.format(type(exc).__name__, exc),
                tb=traceback.format_exc())
            print(paint(failure.render(), 'red'), flush=True)
            report = getattr(test_obj, 'report', None) or TestReport(name)
            report.checks_run += 1
            report.failures.append(failure)

        reports.append(report)
        failures.extend(report.failures)

    print_summary(reports, failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
