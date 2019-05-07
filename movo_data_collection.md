
# Documentation
### Documentation for collecting discrete (image, action) data using Movo

#### I forked [this repo](https://github.com/ros-teleop/teleop_twist_keyboard) and modified it to save data for me.

##### 1. Download

1. Download the [submodule](Training_data_collector_movo/teleop_twist_keyboard/) to `/src` of your catkin workspace.

##### 2.  Building

Navigate to the root of that workspace and run
```
$ catkin_make
```

#### Running

First start **Movo** and set up `ROS_MASTER_URI` and `ROS_IP`- consult the author for this.

To run the `teleop` node, make sure you've sourced your workspace by running
```
$ source <your workspace root>/devel/setup.bash
```
then run on separate terminal
```
$ rosrun teleop_twist_keyboard teleop_twist_keyboard.py _speed:=0.3
```

from `Training_data_collector_movo/teleop_twist_keyboard/`

- Use `i` to go `forward`, `j` to turn `left` and `l` to turn `right`.
- `scp` your data from **Movo**. (Consult the authors).
