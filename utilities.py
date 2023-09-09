import numpy as np
import sympy as sp


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
        # 坡面法向量
        slope_normal_vec = sp.Matrix([sp.sin(self.slope_angle), 0, sp.cos(self.slope_angle)]).evalf(self.precision)
        # 坡面
        self.slope = sp.Plane(self.origin, normal_vector=slope_normal_vec)

        return self.slope
    
    def get_direction_vec(self):
        # 船的方向向量 - direction vector
        self.direction_vec = sp.Matrix([sp.cos(self.direction_angle), sp.sin(self.direction_angle), 0]).evalf(self.precision)

        return self.direction_vec
    
    def get_side_vec(self):
        # 船的方向向量在x-y平面上旋转90度，得到船的侧向向量 - side vector
        self.side_vec = sp.Matrix([-sp.sin(self.direction_angle), sp.cos(self.direction_angle), 0]).evalf(self.precision)

        return self.side_vec

    def get_detect_line(self):
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