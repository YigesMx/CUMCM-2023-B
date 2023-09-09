from typing import List, Tuple

import numpy as np
import sympy as sp


class Field:

    def __init__(self, length, width, center_depth, slope_angle, precision=20):
        self.length = length
        self.width = width
        self.center_depth = center_depth
        self.slope_angle = slope_angle

        self.precision = precision

        # 原点为中心位置在海底坡面的投影点
        self.origin = sp.Matrix([0, 0, 0])
        self.i_hat = sp.Matrix([1, 0, 0]) # x unit, also slope direction's projection onto the x-y plane
        self.j_hat = sp.Matrix([0, 1, 0]) # y unit
        self.k_hat = sp.Matrix([0, 0, 1]) # z unit

        # 坡面法向量
        self.slope_normal_vec = sp.Matrix([sp.sin(self.slope_angle), 0, sp.cos(self.slope_angle)]).evalf(self.precision)
        # 坡面
        self.slope_surface = sp.Plane(self.origin, normal_vector=self.slope_normal_vec)

        # 海平面 sea_level
        self.sea_level = sp.Plane(sp.Point3D([0, 0, center_depth]), normal_vector=self.k_hat)

    def set_slope_surface(self, surface):
        self.slope_surface = surface

    def get_depth(self, x, y):
        
        # xy平面上的点
        xy_point = sp.Point3D([x, y, 0])
        # xy平面上的点到海平面的铅垂线
        plumb_line = self.sea_level.perpendicular_line(xy_point)
        # xy平面上的点到海平面的铅垂线与海平面的交点
        depth_point = self.slope_surface.intersection(plumb_line)
        assert len(depth_point) == 1
        depth_point = depth_point[0]

        return sp.Abs(self.center_depth-depth_point.z.evalf(self.precision))


class Detector:

    def __init__(self, field, x, y, direction_angle, detector_angle, precision=20):
        self.field = field
        self.loc_depth = field.get_depth(x, y)
        self.loc = sp.Matrix([x, y, field.center_depth])
        self.depth_point = sp.Matrix([x, y, field.center_depth-self.loc_depth])
        self.direction_angle = direction_angle
        self.detector_angle = detector_angle
        self.precision = precision
    
    def get_direction_vec(self):
        if hasattr(self, 'direction_vec'):
            return self.direction_vec
        # 船的方向向量 - direction vector
        self.direction_vec = sp.Matrix([sp.cos(self.direction_angle), sp.sin(self.direction_angle), 0]).evalf(self.precision)

        return self.direction_vec
    
    def get_side_vec(self):
        if hasattr(self, 'side_vec'):
            return self.side_vec
        # 船的方向向量在x-y平面上旋转90度，得到船的侧向向量 - side vector
        self.side_vec = sp.Matrix([-sp.sin(self.direction_angle), sp.cos(self.direction_angle), 0]).evalf(self.precision)

        return self.side_vec
    
    def get_detect_line(self):
        if hasattr(self, 'detect_line_1') and hasattr(self, 'detect_line_2'):
            return self.detect_line_1, self.detect_line_2

        if not hasattr(self, 'side_vec'):
            self.get_side_vec()
        # 探测器单边检测角度
        detector_angle_half = self.detector_angle/2
        # 探测器检测扇形与x-y平面的两个交点距离探测器的水平向量
        detect_point_xy_1 = self.depth_point + (self.side_vec * sp.tan(detector_angle_half) * self.loc_depth).evalf(self.precision)
        detect_point_xy_2 = self.depth_point + (-self.side_vec * sp.tan(detector_angle_half) * self.loc_depth).evalf(self.precision)
        # 检测扇形的两个边沿
        self.detect_line_1 = sp.Line(self.loc, detect_point_xy_1)
        self.detect_line_2 = sp.Line(self.loc, detect_point_xy_2)

        return self.detect_line_1, self.detect_line_2
    
    def get_detect_point(self):
        if hasattr(self, 'detect_point_1') and hasattr(self, 'detect_point_2'):
            return self.detect_point_1, self.detect_point_2

        if not hasattr(self, 'detect_line_1') or not hasattr(self, 'detect_line_2'):
            self.get_detect_line()

        # 检测扇形的两个边沿与坡面的交点
        self.detect_point_1 = self.field.slope_surface.intersection(self.detect_line_1)
        self.detect_point_2 = self.field.slope_surface.intersection(self.detect_line_2)
        assert len(self.detect_point_1) == 1 and len(self.detect_point_2) == 1
        self.detect_point_1 = self.detect_point_1[0].evalf(self.precision)
        self.detect_point_2 = self.detect_point_2[0].evalf(self.precision)

        return self.detect_point_1, self.detect_point_2
    
    def get_scan_segment(self): # 包括与其紧密相关的属性
        if hasattr(self, 'scan_segment'):
            return self.scan_segment

        if not hasattr(self, 'detect_point_1') or not hasattr(self, 'detect_point_2'):
            self.get_detect_point()
        # 检测扇形的两个边沿与坡面的交点的连线
        self.scan_segment = sp.Segment(self.detect_point_1, self.detect_point_2)

        # detect_scan_line与x-y平面的夹角
        self.scan_segment_angle_xy = (sp.Abs(sp.pi/2 - sp.Abs(self.scan_segment.angle_between(sp.Line(self.field.origin, self.field.k_hat))))).evalf(self.precision)
        # 检测扇形的两个边沿与坡面的交点的连线的长度
        self.scan_segment_length = self.scan_segment.length
        # detect_scan_line_length在x-y平面上的投影长度
        self.scan_segment_length_proj_xy = (self.scan_segment_length * sp.cos(self.scan_segment_angle_xy)).evalf(self.precision)
        
        return self.scan_segment

class MeasuringLine:

    def __init__(self, field, x, y, direction_angle, forward_length, detector_angle, precision=20):
        self.field = field
        self.detector_0 = Detector(field, x, y, direction_angle, detector_angle, precision)
        x_offset = forward_length * sp.cos(direction_angle) # x方向偏移量
        y_offset = forward_length * sp.sin(direction_angle) # y方向偏移量
        self.detector_1 = Detector(field, x+x_offset, y+y_offset, direction_angle, detector_angle, precision)
        self.precision = precision

    def get_measuring_line(self): # 包括与其紧密相关的属性
        if hasattr(self, 'measuring_line'):
            return self.measuring_line
        
        # 测线(过原点的平行线) 与 测线所在竖直平面
        self.measuring_line = sp.Line(self.detector_0.loc, self.detector_1.loc)
        self.measuring_line_plane = sp.Plane(self.origin, normal_vector=self.detector_0.get_side_vec())

        # 测线平面与坡面的交线
        self.measuring_slope_line = self.measuring_line_plane.intersection(self.field.slope_surface)
        assert len(self.measuring_slope_line) == 1
        self.measuring_slope_line = self.measuring_slope_line[0]

        # measuring_slope_line与x-y平面的夹角
        self.measuring_slope_line_angle_xy = (sp.Abs(sp.pi/2 - sp.Abs(self.measuring_slope_line.angle_between(sp.Line(self.field.origin, self.field.k_hat))))).evalf(self.precision)

        return self.measuring_line
    


class SingleDetector:

    def __init__(self, center_depth, slope_angle, direction_angle, detector_angle, precision=20):
        self.center_depth = center_depth
        self.slope_angle = slope_angle
        self.direction_angle = direction_angle
        self.detector_angle = detector_angle
        self.precision = precision

        # 原点为当前位置在海底坡面的投影点
        self.origin = sp.Matrix([0, 0, 0])
        self.i_hat = sp.Matrix([1, 0, 0]) # x unit, also slope direction's projection onto the x-y plane
        self.j_hat = sp.Matrix([0, 1, 0]) # y unit
        self.k_hat = sp.Matrix([0, 0, 1]) # z unit

        self.detector_loc = sp.Matrix([0, 0, self.center_depth])

    
    def get_slope(self):
        if hasattr(self, 'slope'):
            return self.slope
        # 坡面法向量
        slope_normal_vec = sp.Matrix([sp.sin(self.slope_angle), 0, sp.cos(self.slope_angle)]).evalf(self.precision)
        # 坡面
        self.slope = sp.Plane(self.origin, normal_vector=slope_normal_vec)

        return self.slope
    
    def get_direction_vec(self):
        if hasattr(self, 'direction_vec'):
            return self.direction_vec
        # 船的方向向量 - direction vector
        self.direction_vec = sp.Matrix([sp.cos(self.direction_angle), sp.sin(self.direction_angle), 0]).evalf(self.precision)

        return self.direction_vec
    
    def get_side_vec(self):
        if hasattr(self, 'side_vec'):
            return self.side_vec
        # 船的方向向量在x-y平面上旋转90度，得到船的侧向向量 - side vector
        self.side_vec = sp.Matrix([-sp.sin(self.direction_angle), sp.cos(self.direction_angle), 0]).evalf(self.precision)

        return self.side_vec

    def get_detect_line(self):
        if hasattr(self, 'detect_line_1') and hasattr(self, 'detect_line_2'):
            return self.detect_line_1, self.detect_line_2

        if not hasattr(self, 'side_vec'):
            self.get_side_vec()
        # 探测器单边检测角度
        detector_angle_half = self.detector_angle/2
        # 探测器检测扇形与x-y平面的两个交点
        detect_point_xy_1 = (self.side_vec * sp.tan(detector_angle_half) * self.center_depth).evalf(self.precision)
        detect_point_xy_2 = (-self.side_vec * sp.tan(detector_angle_half) * self.center_depth).evalf(self.precision)
        # 检测扇形的两个边沿
        self.detect_line_1 = sp.Line(self.detector_loc, detect_point_xy_1)
        self.detect_line_2 = sp.Line(self.detector_loc, detect_point_xy_2)

        return self.detect_line_1, self.detect_line_2
    
    def get_detect_point(self):
        if hasattr(self, 'detect_point_1') and hasattr(self, 'detect_point_2'):
            return self.detect_point_1, self.detect_point_2

        if not hasattr(self, 'detect_line_1') or not hasattr(self, 'detect_line_2'):
            self.get_detect_line()
        if not hasattr(self, 'slope'):
            self.get_slope()
        # 检测扇形的两个边沿与坡面的交点
        self.detect_point_1 = self.slope.intersection(self.detect_line_1)
        self.detect_point_2 = self.slope.intersection(self.detect_line_2)
        assert len(self.detect_point_1) == 1 and len(self.detect_point_2) == 1
        self.detect_point_1 = self.detect_point_1[0].evalf(self.precision)
        self.detect_point_2 = self.detect_point_2[0].evalf(self.precision)

        return self.detect_point_1, self.detect_point_2
    
    def get_scan_segment(self): # 包括与其紧密相关的属性
        if hasattr(self, 'scan_segment'):
            return self.scan_segment

        if not hasattr(self, 'detect_point_1') or not hasattr(self, 'detect_point_2'):
            self.get_detect_point()
        # 检测扇形的两个边沿与坡面的交点的连线
        self.scan_segment = sp.Segment(self.detect_point_1, self.detect_point_2)

        # detect_scan_line与x-y平面的夹角
        self.scan_segment_angle_xy = (sp.Abs(sp.pi/2 - sp.Abs(self.scan_segment.angle_between(sp.Line(self.origin, self.k_hat))))).evalf(self.precision)
        # 检测扇形的两个边沿与坡面的交点的连线的长度
        self.scan_segment_length = self.scan_segment.length
        # detect_scan_line_length在x-y平面上的投影长度
        self.scan_segment_length_proj_xy = (self.scan_segment_length * sp.cos(self.scan_segment_angle_xy)).evalf(self.precision)
        
        return self.scan_segment
    
    def get_measuring_line(self): # 包括与其紧密相关的属性
        if hasattr(self, 'measuring_line'):
            return self.measuring_line

        if not hasattr(self, 'direction_vec'):
            self.get_direction_vec()
        if not hasattr(self, 'side_vec'):
            self.get_side_vec()
        if not hasattr(self, 'slope'):
            self.get_slope()
        
        # 测线(过原点的平行线) 与 测线所在竖直平面
        self.measuring_line = sp.Line(self.origin, self.direction_vec)
        self.measuring_line_plane = sp.Plane(self.origin, normal_vector=self.side_vec)

        # 测线平面与坡面的交线
        self.measuring_slope_line = self.measuring_line_plane.intersection(self.slope)
        assert len(self.measuring_slope_line) == 1
        self.measuring_slope_line = self.measuring_slope_line[0]

        # measuring_slope_line与x-y平面的夹角
        self.measuring_slope_line_angle_xy = (sp.Abs(sp.pi/2 - sp.Abs(self.measuring_slope_line.angle_between(sp.Line(self.origin, self.k_hat))))).evalf(self.precision)

        return self.measuring_line

    # def detect(self):
    #     # 坡面法向量
    #     slope_normal_vec = sp.Matrix([sp.sin(self.slope_angle), 0, sp.cos(self.slope_angle)])
    #     # 坡面
    #     self.slope = sp.Plane(self.origin, normal_vector=slope_normal_vec)

    #     # 船的方向向量 - direction vector
    #     self.direction_vec = sp.Matrix([sp.cos(self.direction_angle), sp.sin(self.direction_angle), 0])
    #     # 船的方向向量在x-y平面上旋转90度，得到船的侧向向量 - side vector
    #     self.side_vec = sp.Matrix([-sp.sin(self.direction_angle), sp.cos(self.direction_angle), 0])

    #     # 探测器单边检测角度
    #     detector_angle_half = self.detector_angle/2
    #     # 探测器检测扇形与x-y平面的两个交点
    #     detect_point_xy_1 = self.side_vec * sp.tan(detector_angle_half) * self.center_depth
    #     detect_point_xy_2 = -self.side_vec * sp.tan(detector_angle_half) * self.center_depth
    #     # 检测扇形的两个边沿
    #     self.detect_line_1 = sp.Line(self.detector_loc, detect_point_xy_1)
    #     self.detect_line_2 = sp.Line(self.detector_loc, detect_point_xy_2)
    #     # 检测扇形的两个边沿与坡面的交点
    #     self.detect_point_1 = self.slope.intersection(self.detect_line_1)
    #     self.detect_point_2 = self.slope.intersection(self.detect_line_2)
    #     assert len(self.detect_point_1) == 1 and len(self.detect_point_2) == 1
    #     self.detect_point_1 = self.detect_point_1[0]
    #     self.detect_point_2 = self.detect_point_2[0]

    #     # 检测扇形的两个边沿与坡面的交点的连线
    #     self.scan_segment = sp.Segment(self.detect_point_1, self.detect_point_2)
    #     # detect_scan_line与x-y平面的夹角
    #     self.scan_segment_angle_xy = sp.Abs(sp.pi/2 - sp.Abs(self.scan_segment.angle_between(sp.Line(self.origin, self.k_hat))))

    #     # 检测扇形的两个边沿与坡面的交点的连线的长度
    #     self.scan_segment_length = self.scan_segment.length
    #     # detect_scan_line_length在x-y平面上的投影长度
    #     self.scan_segment_length_proj_xy = self.scan_segment_length * sp.cos(self.scan_segment_angle_xy)