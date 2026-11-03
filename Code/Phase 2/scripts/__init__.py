"""Phase 2 scripts package.

Present so that ``import scripts.<module>`` resolves to this directory even
on systems where a conflicting ``scripts`` package (e.g. ROS's gazebo_ros)
sits earlier on sys.path. Entry points still add ``scripts/`` to sys.path
and import ``_train_common`` as a top-level module; this file does not
change that.
"""
