#!/usr/bin/env python3
"""
Kiem tra sap xep sys.path de uu tien Python 3 truoc ROS Python 2.7.
"""
import sys
import unittest


class TestPathOrdering(unittest.TestCase):
    def test_sys_path_ordering(self):
        """Kiem tra logic tach va sap xep sys.path."""
        test_paths = ['/opt/ros/melodic/lib/python2.7/dist-packages', '/usr/lib/python3/dist-packages']
        py3_paths = [p for p in test_paths if 'python2.7' not in p]
        py2_paths = [p for p in test_paths if 'python2.7' in p]
        reordered = py3_paths + py2_paths

        self.assertNotIn('python2.7', reordered[0])
        self.assertIn('python2.7', reordered[-1])


if __name__ == '__main__':
    unittest.main()
