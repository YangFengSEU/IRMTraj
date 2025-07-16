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

logger = logging.getLogger(__name__)


_PathLike = Union[str, "os.PathLike[str]"]


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    
    return (b*b-a*a-c*c)/(-2*a*c+0.00001)


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
   
def append_additional_key_value_pair(lane_obj: MutableMapping[str, Any], way_field: List[Tuple[str, str]]) -> None:
    """
    Key name was either 'predecessor' or 'successor', for which we can have multiple.
    Thus we append them to a list. They should be integers, as lane IDs.

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = int(way_field[1][1])
    lane_obj.setdefault(k, []).append(v)


def append_unique_key_value_pair(lane_obj: MutableMapping[str, Any], way_field: List[Tuple[str, str]]) -> None:
    """
    For the following types of Way "tags", the key, value pair is defined only once within
    the object:
        - has_traffic_control, turn_direction, is_intersection, l_neighbor_id, r_neighbor_id

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = way_field[1][1]
    lane_obj[k] = v

def convert_node_id_list_to_xy(node_id_list: List[int], all_graph_nodes: Mapping[int, Node]) -> np.ndarray:
    """
    convert node id list to centerline xy coordinate

    Args:
       node_id_list: list of node_id's
       all_graph_nodes: dictionary mapping node_ids to Node

    Returns:
       centerline
    """
    num_nodes = len(node_id_list)
    centerline = np.zeros((num_nodes, 2))
    for i, node_id in enumerate(node_id_list):
        centerline[i] = np.array([all_graph_nodes[node_id].x, all_graph_nodes[node_id].y])

    return centerline

def extract_way_segment_from_ET_element(
    child: ET.Element
) -> Tuple[WaySeg, int]:
    """
    build a lane segment from an XML element. A lane segment is equivalent
    to a "Way" in our XML file.
    The relevant XML data might resemble::
      </way>
      <way id="10131" visible="true" version="1">
        <nd ref="1603" />
        <nd ref="1646" />
        <nd ref="1365" />
        <tag k="subtype" v="dashed" />
        <tag k="type" v="line_thin" />

    Args:
        child: xml.etree.ElementTree element
        all_graph_nodes

    Returns:
        lane_segment: LaneSegment object
        lane_id
    
    """
    way_obj: Dict[str, Any] = {}
    way_id = int(child.get('id'))
    node_id_list: List[int] = []
    way_obj["subtype"]=None
    way_obj["lane_change"]=False
    for element in child:
        # The cast on the next line is the result of a typeshed bug.  This really is a List and not a ItemsView.
        way_field = cast(List[Tuple[str, str]], list(element.items()))#强制的类型转换
        field_name = way_field[0][0]
        if field_name == "k":
            key = way_field[0][1]
            if key =="type":
                way_obj["type"]=way_field[1][1]
            elif key =="subtype":
                way_obj["subtype"]=way_field[1][1]
            elif key =="lane_change" and way_field[1][1]=="yes":
                way_obj["lane_change"]=True
        elif field_name == "ref":
            node_id_list.append(int(way_field[0][1]))
    
    way_seg = WaySeg(
        way_id,
        node_id_list,
        way_obj["type"],
        way_obj["subtype"],
        way_obj["lane_change"],
    )
    return way_seg, way_id

def extract_relation_segment_from_ET_element(
    child: ET.Element
) -> Tuple[RalationSeg, int]:
    """
    build a lane segment from an XML element. A lane segment is equivalent
    to a "Way" in our XML file.
    The relevant XML data might resemble::
      <relation id="30065" visible="true" version="1">
        <member type="way" ref="10057" role="left" />
        <member type="way" ref="10098" role="right" />
        <member type="relation" ref="50000" role="regulatory_element" />
        <tag k="location" v="urban" />
        <tag k="one_way" v="yes" />
        <tag k="region" v="us-ca" />
        <tag k="subtype" v="road" />
        <tag k="type" v="lanelet" />

    Args:
        child: xml.etree.ElementTree element
        all_graph_nodes

    Returns:
        lane_segment: LaneSegment object
        lane_id
    """
    relation_obj: Dict[str, Any] = {}
    relation_id = int(child.get('id'))
    way_id_list: List[int] = []
    relation_obj["oneway"]=False
    for element in child:
        # The cast on the next line is the result of a typeshed bug.  This really is a List and not a ItemsView.
        rela_field = cast(List[Tuple[str, str]], list(element.items()))#强制的类型转换
        field_name = rela_field[0][0]
        if field_name == "k":
            key = rela_field[0][1]
            if key =="location":
                relation_obj["location"]=rela_field[1][1]
            elif key =="oneway" and rela_field[1][1]=="yes":
                relation_obj["oneway"]=True
            elif key =="subtype":
                relation_obj["subtype"]=rela_field[1][1]
            elif key =="type":
                relation_obj["type"]=rela_field[1][1]
                
        elif field_name == "type":
            key = rela_field[0][1]
            if key =="way":
                if rela_field[2][1]=="left":
                    relation_obj["left"]=int(rela_field[1][1])
                elif rela_field[2][1]=="right":
                    relation_obj["right"]=int(rela_field[1][1]) 
    
    ralationSeg = RalationSeg(
        relation_id,
        relation_obj["left"],
        relation_obj["right"],
        relation_obj["location"],
        relation_obj["oneway"],
        relation_obj["subtype"],
        relation_obj["type"],
        None,
    )
    return ralationSeg, relation_id

def extend_node_of_line(line,num):
    reline=line
    while reline.shape[0]<num:
        a=np.zeros([reline.shape[0]-1])
        for i in range(reline.shape[0]-1):
            a[i]=(reline[i,0]-reline[i+1,0])*(reline[i,0]-reline[i+1,0])+(reline[i,1]-reline[i+1,1])*(reline[i,1]-reline[i+1,1])
        maxid=np.argmax(a)
        x=(reline[maxid,0]+reline[maxid+1,0])/2
        y=(reline[maxid,1]+reline[maxid+1,1])/2
        reline = np.insert(reline,(maxid+1)*2,[x,y]).reshape(-1,2)
        
    return reline

def build_lane_obj(relation_dict,way_objs,point_dict)-> Tuple[WaySeg, int]:
    #1.确定起点和终点，顺便建立起点终点表
    #2.算中心线
    #3.算前驱后继
    #4. 算左右邻居
    lane_objs={}
    preandsuc_list={}
    nei_list={}
    for rela_id,rela_obj in relation_dict.items():
        left=way_objs[rela_obj.leftid].ndid
        right=way_objs[rela_obj.rightid].ndid
        
        #矫正两边线的方向
        checkx=(point_dict[left[0]].x-point_dict[left[-1]].x)
        checky=(point_dict[left[0]].y-point_dict[left[-1]].y)
        
        if abs(checkx)>=abs(checky):
            if checkx*(point_dict[right[0]].x-point_dict[right[-1]].x)<0:
                left=list(reversed(left))
        else:
            if checky*(point_dict[right[0]].y-point_dict[right[-1]].y)<0:
                left=list(reversed(left))
        #维护两个表，前驱后继表和邻居表
        preandsuc_list.setdefault((left[0],right[0],0),[]).append(rela_id)
        preandsuc_list.setdefault((left[-1],right[-1],1),[]).append(rela_id)
        
        nei_list[(rela_obj.leftid,0)]=rela_id
        nei_list[(rela_obj.rightid,1)]=rela_id
        
        #计算中心线
        leftline = convert_node_id_list_to_xy(left, point_dict)
        rightline = convert_node_id_list_to_xy(right, point_dict)
        
        if leftline.shape[0] > rightline.shape[0]:
            rightline = extend_node_of_line(rightline,leftline.shape[0])
        else:
            leftline = extend_node_of_line(leftline,rightline.shape[0])
        
        for i in range(rightline.shape[0]):
            rightline[i,0]=(rightline[i,0]+leftline[i,0])/2
            rightline[i,1]=(rightline[i,1]+leftline[i,1])/2
            
        rela_obj.centerline=rightline
    
    for rela_id,rela_obj in relation_dict.items():
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
        """
        left=way_objs[rela_obj.leftid].ndid
        right=way_objs[rela_obj.rightid].ndid
        
        #矫正两边线的方向
        checkx=(point_dict[left[0]].x-point_dict[left[-1]].x)
        checky=(point_dict[left[0]].y-point_dict[left[-1]].y)
        
        if abs(checkx)>=abs(checky):
            if checkx*(point_dict[right[0]].x-point_dict[right[-1]].x)<0:
                left=list(reversed(left))
        else:
            if checky*(point_dict[right[0]].y-point_dict[right[-1]].y)<0:
                left=list(reversed(left))
          
        l_neighbor_id = nei_list[(rela_obj.leftid,1)] if (rela_obj.leftid,1) in nei_list else None
        r_neighbor_id = nei_list[(rela_obj.rightid,0)] if (rela_obj.rightid,0) in nei_list else None

        pre=[]
        if (left[-1],right[-1],0) in preandsuc_list:
            pre=pre+preandsuc_list[(left[-1],right[-1],0)]
        if (right[-1],left[-1],0) in preandsuc_list:
            pre=pre+preandsuc_list[(right[-1],left[-1],0)]
        
        suc=[]
        if (left[0],right[0],1) in preandsuc_list:
            suc=suc+preandsuc_list[(left[0],right[0],1)]
        if (right[0],left[0],1) in preandsuc_list:
            suc=suc+preandsuc_list[(right[0],left[0],1)]
            
        
        #滤去锐角的前驱和后继

        predecessors=[]
        for preid in pre:
            lanepre=relation_dict[preid].centerline 
            lanenow=rela_obj.centerline[-2:]       
            costheta=cal_ang((lanepre[-1][0], lanepre[-1][1]), (lanepre[0][0], lanepre[0][1]), (lanenow[0][0], lanenow[0][1]))
            
            if costheta<0:
                predecessors=predecessors+[preid]
        

        successors=[]
        for sucid in suc:
            lanepre=relation_dict[sucid].centerline
            lanenow=rela_obj.centerline[0:2]          
            costheta=cal_ang((lanepre[0][0], lanepre[0][1]), (lanepre[-1][0], lanepre[-1][1]), (lanenow[1][0], lanenow[1][1]))
            
            if costheta<0:
                successors=successors+[sucid]
        if successors ==[]:
            successors=None
        
        
        lane_obj=LaneSegment(
                rela_id,
                False,
                None,
                False,
                l_neighbor_id,
                r_neighbor_id,
                predecessors,
                successors,
                rela_obj.centerline,
            )
        lane_objs[rela_id]=lane_obj
    return lane_objs
        
def get_jsonfile_and_npfile(relation_dict,way_objs,point_dict):
    tableid_to_lane={}
    halluc_bbox_table=np.zeros([len(relation_dict),4])
    minid=40000
    for laneid,rela_obj in relation_dict.items():       
        if laneid<minid:
            minid=laneid
        tableid_to_lane[str(laneid-minid)]=laneid
        
        left=way_objs[rela_obj.leftid].ndid
        right=way_objs[rela_obj.rightid].ndid
        leftline = convert_node_id_list_to_xy(left, point_dict)
        rightline = convert_node_id_list_to_xy(right, point_dict)
        
        line=np.concatenate((leftline,rightline)).T
        
        xmax=max(line[0])
        xmin=min(line[0])
        
        ymax=max(line[1])
        ymin=min(line[1])
        
        halluc_bbox_table[laneid-minid]=[xmin,ymin,xmax,ymax]
    
    return tableid_to_lane,halluc_bbox_table
        

def load_lane_segments_from_xml(map_fpath: _PathLike,viz: bool = False) -> Mapping[int, LaneSegment]:
    """
    Load lane segment object from xml file

    Args:
       map_fpath: path to xml file

    Returns:
       lane_objs: List of LaneSegment objects
    """
    tree = ET.parse(os.fspath(map_fpath))
    root = tree.getroot()
    projector = LL2XYProjector(0.0, 0.0)
    logger.info(f"Loaded root: {root.tag}")

    way_objs = {}
    point_dict = dict()
    relation_dict={}
    lane_objs = {}
    for node in root.findall("node"):
        point = Node()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point.id=int(node.get('id'))
        point_dict[int(node.get('id'))] = point
    # all children are either Nodes or Ways
    
    for way in root.findall("way"):
        way_obj, way_id = extract_way_segment_from_ET_element(way)
        way_objs[way_id] = way_obj

    for relation in root.findall("relation"):
        for tag in relation.findall("tag"):
            if tag.get("k") == "type":
                roadtype=tag.get("v")
        if roadtype!="lanelet":
            continue
        relation_obj, relation_id = extract_relation_segment_from_ET_element(relation)
        relation_dict[relation_id] = relation_obj   
        
    lane_objs = build_lane_obj(relation_dict,way_objs,point_dict)
    if viz==True:
        for laneid,lane in lane_objs.items():
            x=lane.centerline[:,0].T
            y=lane.centerline[:,1].T
            plt.plot(x,y,linewidth=1)
        plt.show()
        
    tableid_to_lane={}
    halluc_bbox_table=np.zeros([len(relation_dict),4])
    tableid_to_lane,halluc_bbox_table = get_jsonfile_and_npfile(relation_dict,way_objs,point_dict)
      
    filepath=os.path.dirname(map_fpath)
    base = os.path.basename(map_fpath).split(".")
    np.save(filepath+"/"+base[0]+"_halluc_bbox_table.npy",halluc_bbox_table)
    
    with open(filepath+"/"+base[0]+"_tableid_to_lane.json",'w') as f:
        json.dump(tableid_to_lane,f)
    

    return lane_objs
