#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union, cast
import pyproj
import math
import matplotlib.pyplot as plt
import json

import numpy as np

class Node:
    def __init__(self):
        self.id = None
        self.x = None
        self.y = None
        
class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)#从经纬度转换成坐标系

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]
        
class LaneSegment:
    def __init__(
        self,
        id: int,
        has_traffic_control: bool,
        turn_direction: str,
        is_intersection: bool,
        l_neighbor_id: Optional[int],
        r_neighbor_id: Optional[int],
        predecessors: List[int],
        successors: Optional[List[int]],
        centerline: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            has_traffic_control:
            turn_direction: 'RIGHT', 'LEFT', or 'NONE'
            is_intersection: Whether or not this lane segment is an intersection
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            predecessors: The IDs of the lane segments that come after this one
            successors: The IDs of the lane segments that come before this one.
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.is_intersection = is_intersection
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline
        
class RalationSeg:
    def __init__(
        self,
        id: int,
        leftid: int,
        rightid: int,
        location: str,
        one_way: bool,
        subtype: str,
        typename: str,
        centerline: Optional[np.ndarray],
    ) -> None:
        """Initialize the lanelet,prepare to transfer lanelet into the laneseg.
        """
        self.id = id
        self.leftid = leftid
        self.rightid = rightid

        self.location = location
        self.one_way = one_way
        self.subtype = subtype
        self.typename = typename
        self.centerline=centerline

class WaySeg:
    def __init__(
        self,
        id: int,
        ndid: [List[int]],
        typename:str,
        subtype: str,
        lane_change:bool,
    ) -> None:
        """Initialize the lanelet,prepare to transfer lanelet into the laneseg.
        """
        self.id = id
        self.ndid = ndid
        self.typename = typename
        self.subtype = subtype
        self.lane_change = lane_change
   

        


