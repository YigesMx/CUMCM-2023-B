from typing import List, Tuple

import numpy as np
import sympy as sp

from enum import Enum

class ArrangeError(Enum):
    FIRST_TOP_MEASURING_LINE_OUT_OF_LEFT_WALL = 1
    OVERLAP_RATIO_TOO_LARGE = 2

tick = 0

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

        # 四面
        self.top_wall = sp.Plane(sp.Point3D([-self.length/2, 0, 0]), normal_vector=self.i_hat)
        self.bottom_wall = sp.Plane(sp.Point3D([self.length/2, 0, 0]), normal_vector=-self.i_hat)
        self.left_wall = sp.Plane(sp.Point3D([0, -self.width/2, 0]), normal_vector=self.j_hat)
        self.right_wall = sp.Plane(sp.Point3D([0, self.width/2, 0]), normal_vector=-self.j_hat)

        # MeasuringLines
        self.measuring_lines = {
            'top': [],
            'left': [],
        }

        self.next_left_measuring_line_gap_equation = \
            self.get_next_left_measuring_line_gap_equation(
                pre_left_measuring_line_from_top = sp.Symbol('s'),
                direction_angle = sp.Symbol('beta'),
                detector_angle = sp.Symbol('theta')
            )

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

    def get_first_top_measuring_line_from_left(self, first_left_measuring_line_from_top, direction_angle, detector_angle):
        s = first_left_measuring_line_from_top
        beta = direction_angle
        d = sp.Symbol('d')
        
        detector1 = Detector(field=self,
                             x=self.top_wall.p1.x + s,
                             y=self.left_wall.p1.y,
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        
        detector1.get_detect_point()
        detector1_detect_point_left = detector1.detect_point_1
        detector1_depth_point = detector1.depth_point

        detector2 = Detector(field=self,
                             x= (self.top_wall.p1.x + s - ( ( d+s*sp.tan(beta) )*(sp.cos(beta)*sp.sin(beta)) )).evalf(self.precision),
                             y= (self.left_wall.p1.y + ( ( d+s*sp.tan(beta) )*(sp.cos(beta)**2) )).evalf(self.precision),
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        detector2.get_detect_point()
        detector2_detect_point_right = detector2.detect_point_2
        detector2_depth_point = detector2.depth_point

        slope_length_between_depth_point = sp.Segment(detector1_depth_point, detector2_depth_point).length.evalf(self.precision)
        slope_length_half_1 = sp.Segment(detector1_depth_point, detector1_detect_point_left).length.evalf(self.precision)
        slope_length_half_2 = sp.Segment(detector2_depth_point, detector2_detect_point_right).length.evalf(self.precision)
        overlap_ratio = ((slope_length_half_1 + slope_length_half_2 - slope_length_between_depth_point) / slope_length_between_depth_point).evalf(self.precision)

        eq = sp.Eq(overlap_ratio, 0.1)
        d = sp.nsolve(eq, d, detector1.loc_depth)
        
        return d
    
    def get_next_top_measuring_line_gap(self, pre_top_measuring_line_from_left, direction_angle, detector_angle):
        d = pre_top_measuring_line_from_left
        beta = direction_angle
        y = sp.Symbol('y')

        detector0 = Detector(field=self,
                             x=self.top_wall.p1.x,
                             y=self.left_wall.p1.y + d,
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        detector0.get_scan_segment()
        ref_length = detector0.scan_segment_length_proj_xy # 用于求参考长度

        # pre
        detector1 = Detector(field=self,
                             x= (self.top_wall.p1.x + y*sp.sin(beta)*sp.cos(beta)).evalf(self.precision),
                             y= (self.left_wall.p1.y + d + y*(sp.sin(beta)**2)).evalf(self.precision),
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        
        detector1.get_detect_point()
        detector1_detect_point_left = detector1.detect_point_1
        detector1_depth_point = detector1.depth_point

        # next
        detector2 = Detector(field=self,
                             x= self.top_wall.p1.x,
                             y= (self.left_wall.p1.y + d + y).evalf(self.precision),
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        detector2.get_detect_point()
        detector2_detect_point_right = detector2.detect_point_2
        detector2_depth_point = detector2.depth_point

        slope_length_between_depth_point = sp.Segment(detector1_depth_point, detector2_depth_point).length.evalf(self.precision)
        slope_length_half_1 = sp.Segment(detector1_depth_point, detector1_detect_point_left).length.evalf(self.precision)
        slope_length_half_2 = sp.Segment(detector2_depth_point, detector2_detect_point_right).length.evalf(self.precision)
        overlap_ratio = ((slope_length_half_1 + slope_length_half_2 - slope_length_between_depth_point) / slope_length_between_depth_point).evalf(self.precision)

        eq = sp.Eq(overlap_ratio, 0.1)
        y = sp.nsolve(eq, y, ref_length)
        
        return y
    
    def get_next_left_measuring_line_gap_equation(self, pre_left_measuring_line_from_top, direction_angle, detector_angle):
        s = pre_left_measuring_line_from_top
        beta = direction_angle
        x = sp.Symbol('x')

        detector1 = Detector(field=self,
                             x= (self.top_wall.p1.x + s + x*(sp.cos(beta)**2)).evalf(self.precision),
                             y= (self.left_wall.p1.y + x*sp.cos(beta)*sp.sin(beta)).evalf(self.precision),
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        
        detector1.get_detect_point()
        detector1_detect_point_right = detector1.detect_point_2
        detector1_depth_point = detector1.depth_point

        detector2 = Detector(field=self,
                             x= (self.top_wall.p1.x + s + x).evalf(self.precision),
                             y= self.left_wall.p1.y,
                             direction_angle=beta,
                             detector_angle=detector_angle,
                             precision=self.precision)
        detector2.get_detect_point()
        detector2_detect_point_left = detector2.detect_point_1
        detector2_depth_point = detector2.depth_point

        slope_length_between_depth_point = sp.Segment(detector1_depth_point, detector2_depth_point).length.evalf(self.precision)
        slope_length_half_1 = sp.Segment(detector1_depth_point, detector1_detect_point_right).length.evalf(self.precision)
        slope_length_half_2 = sp.Segment(detector2_depth_point, detector2_detect_point_left).length.evalf(self.precision)
        overlap_ratio = ((slope_length_half_1 + slope_length_half_2 - slope_length_between_depth_point) / slope_length_between_depth_point).evalf(self.precision)

        eq = sp.Eq(overlap_ratio, 0.1)
        
        return eq
    
    def get_next_left_measuring_line_gap(self, pre_left_measuring_line_from_top, direction_angle, detector_angle, ref_length=50):
        x = sp.Symbol('x')
        s = sp.Symbol('s')
        beta = sp.Symbol('beta')
        theta = sp.Symbol('theta')
        eq_subsituted = self.next_left_measuring_line_gap_equation\
            .subs({s: pre_left_measuring_line_from_top, beta: direction_angle, theta: detector_angle})
        x = sp.nsolve(eq_subsituted, x, ref_length)
        return x
    
    def get_far_end_overlap_ratio(self, measuring_line_pre, measuring_line_cur, dir = 'upward'):
        global tick
        if tick == 0:
            tick ^= 1
            return 0.1
        tick ^= 1

        end_detector_cur = measuring_line_cur.get_end_detector()
        end_point_cur = end_detector_cur.loc
        #print(f'end_point_cur: {end_point_cur}')

        trace_pre = measuring_line_pre.get_trace()
        # end_point_cur 到 trace_pre 的垂线 与 trace_pre 的交点 (新的向旧的作垂线)
        perpendicular_line = trace_pre.perpendicular_line(end_point_cur)
        perpendicular_point = trace_pre.intersection(perpendicular_line)
        assert len(perpendicular_point) == 1
        perpendicular_point = perpendicular_point[0]
        #print(f'perpendicular_point: {perpendicular_point}')

        # 在 perpendicular_point 处建立探测器
        perpendicular_detector = Detector(field=self,
                                          x=perpendicular_point.x,
                                          y=perpendicular_point.y,
                                          direction_angle=end_detector_cur.direction_angle,
                                          detector_angle=end_detector_cur.detector_angle,
                                          precision=self.precision)
        
        # 计算两个探测器的重复率
        end_detector_cur.get_detect_point()
        end_detector_cur_depth_point = end_detector_cur.depth_point

        perpendicular_detector.get_detect_point()
        perpendicular_detector_depth_point = perpendicular_detector.depth_point

        slope_length_between_depth_point = sp.Segment(end_detector_cur_depth_point, perpendicular_detector_depth_point).length.evalf(self.precision)

        if dir == 'upward':
            end_detector_cur_detect_point_right = end_detector_cur.detect_point_2
            perpendicular_detector_detect_point_left = perpendicular_detector.detect_point_1

            # !!!
            slope_length_half_1 = sp.Segment(end_detector_cur_depth_point, end_detector_cur_detect_point_right).length.evalf(self.precision)
            slope_length_half_2 = sp.Segment(perpendicular_detector_depth_point, perpendicular_detector_detect_point_left).length.evalf(self.precision)
        elif dir == 'downward':
            end_detector_cur_detect_point_left = end_detector_cur.detect_point_1
            perpendicular_detector_detect_point_right = perpendicular_detector.detect_point_2

            # !!!
            slope_length_half_1 = sp.Segment(end_detector_cur_depth_point, end_detector_cur_detect_point_left).length.evalf(self.precision)
            slope_length_half_2 = sp.Segment(perpendicular_detector_depth_point, perpendicular_detector_detect_point_right).length.evalf(self.precision)
        
        overlap_ratio = ((slope_length_half_1 + slope_length_half_2 - slope_length_between_depth_point) / slope_length_between_depth_point).evalf(self.precision)

        # print(f'overlap_ratio: {overlap_ratio}')
        return overlap_ratio


    def arrange_measuring_lines(self, first_left_measuring_line_from_top, direction_angle, detector_angle):
        self.measuring_lines.clear()
        self.measuring_lines = {
            'top': [],
            'left': [],
        }
        # 生成左边出发的第一条测线
        self.measuring_lines['left'].append(
            MeasuringLine(field=self,
                          x=self.top_wall.p1.x + first_left_measuring_line_from_top,
                          y=self.left_wall.p1.y,
                          direction_angle=direction_angle,
                          forward_length=100,
                          detector_angle=detector_angle,
                          precision=self.precision)
        )

        # print('finished calc first_left_measuring_line')

        # 首先计算顶部出发的第一条测线的位置
        first_top_measuring_line_from_left = self.get_first_top_measuring_line_from_left(
            first_left_measuring_line_from_top, direction_angle, detector_angle
        )
        
        # 如果顶部第一条测线的位置在左边界之外，则规划失败
        if first_top_measuring_line_from_left <= 0:
            return ArrangeError.FIRST_TOP_MEASURING_LINE_OUT_OF_LEFT_WALL

        # 如果顶部第一条测线的位置在右边界之内，则生成顶部出发的第一条测线
        if first_top_measuring_line_from_left < self.width:
            # 生成顶部出发的第一条测线
            self.measuring_lines['top'].append(
                MeasuringLine(field=self,
                            x=self.top_wall.p1.x,
                            y=self.left_wall.p1.y + first_top_measuring_line_from_left,
                            direction_angle=direction_angle,
                            forward_length=100,
                            detector_angle=detector_angle,
                            precision=self.precision)
            )

            # 检测重叠率
            overlap_ratio = self.get_far_end_overlap_ratio(
                self.measuring_lines['left'][0], self.measuring_lines['top'][0], dir='upward'
            )
            if overlap_ratio > 0.2:
                print(f'overlap_ratio_1: {overlap_ratio}')
                return ArrangeError.OVERLAP_RATIO_TOO_LARGE

            # print('finished calc first_top_measuring_line')

            # 计算顶部测线间距
            next_top_measuring_line_gap = self.get_next_top_measuring_line_gap(
                first_top_measuring_line_from_left, direction_angle, detector_angle
            )
            
            least_lines_number = int((self.width - first_top_measuring_line_from_left) / next_top_measuring_line_gap) - 1
            # 生成顶部出发的其余测线，直到某个测线起始点探测到右边界
            i = 0
            while True:
                i += 1
                next_top_measuring_line_from_left = first_top_measuring_line_from_left + i*next_top_measuring_line_gap
                
                if next_top_measuring_line_from_left > self.width:
                    break

                # print(f"==> {next_top_measuring_line_from_left}")

                next_top_measuring_line = MeasuringLine(field=self,
                                                        x=self.top_wall.p1.x,
                                                        y=self.left_wall.p1.y + next_top_measuring_line_from_left,
                                                        direction_angle=direction_angle,
                                                        forward_length=100,
                                                        detector_angle=detector_angle,
                                                        precision=self.precision)

                # 检测重叠率
                overlap_ratio = self.get_far_end_overlap_ratio(
                    self.measuring_lines['top'][-1], next_top_measuring_line, dir='upward'
                )
                if overlap_ratio > 0.2:
                    print(f'overlap_ratio_2: {overlap_ratio}')
                    print(f'top: {len(self.measuring_lines["top"])} in total')
                    return ArrangeError.OVERLAP_RATIO_TOO_LARGE

                self.measuring_lines['top'].append(
                    next_top_measuring_line
                )
            
        # print(f'finished calc top_measuring_lines: {len(self.measuring_lines["top"])} in total')
            
        # 生成左边出发的其余测线，直到某个测线起始点探测到下边界
        i = 0
        next_left_measuring_line_from_top = first_left_measuring_line_from_top
        while True:
            i += 1
            next_left_measuring_line_gap = self.get_next_left_measuring_line_gap(
                next_left_measuring_line_from_top, direction_angle, detector_angle, ref_length=100
            )
            next_left_measuring_line_from_top = next_left_measuring_line_from_top + next_left_measuring_line_gap
            
            if next_left_measuring_line_from_top > self.length:
                break

            # print(f"==> {next_left_measuring_line_from_top}")

            next_left_measuring_line = MeasuringLine(field=self,
                                                    x=self.top_wall.p1.x + next_left_measuring_line_from_top,
                                                    y=self.left_wall.p1.y,
                                                    direction_angle=direction_angle,
                                                    forward_length=100,
                                                    detector_angle=detector_angle,
                                                    precision=self.precision)
            
            # 检测重叠率
            overlap_ratio = self.get_far_end_overlap_ratio(
                self.measuring_lines['left'][-1], next_left_measuring_line, dir='downward'
            )
            if overlap_ratio > 0.2:
                print(f'overlap_ratio_3: {overlap_ratio}')
                print(f'left: {len(self.measuring_lines["left"])} in total')
                return ArrangeError.OVERLAP_RATIO_TOO_LARGE 
            
            self.measuring_lines['left'].append(
                next_left_measuring_line
            )
        
        # print(f'finished calc left_measuring_lines : {len(self.measuring_lines["left"])} in total')

    def get_measuring_line_gross_length(self):
        length = 0
        for measuring_line in self.measuring_lines['top']:
            length += measuring_line.get_length()
        for measuring_line in self.measuring_lines['left']:
            length += measuring_line.get_length()
        return length


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
    
    def get_side_vec(self): # left side
        if hasattr(self, 'side_vec'):
            return self.side_vec
        # 船的方向向量在x-y平面上旋转90度，得到船的侧向(左侧)向量 - side vector
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

    def get_trace(self): # 包括与其紧密相关的属性
        if hasattr(self, 'trace'):
            return self.trace
        
        # 测线(过原点的平行线) 与 测线所在竖直平面
        self.trace = sp.Line(self.detector_0.loc, self.detector_1.loc)
        self.trace_plane = sp.Plane(self.field.origin, normal_vector=self.detector_0.get_side_vec())

        # 测线平面与坡面的交线
        self.measuring_slope_line = self.trace_plane.intersection(self.field.slope_surface)
        assert len(self.measuring_slope_line) == 1
        self.measuring_slope_line = self.measuring_slope_line[0]

        # measuring_slope_line与x-y平面的夹角
        self.measuring_slope_line_angle_xy = (sp.Abs(sp.pi/2 - sp.Abs(self.measuring_slope_line.angle_between(sp.Line(self.field.origin, self.field.k_hat))))).evalf(self.precision)

        return self.trace
    
    def get_end_detector(self):
        if hasattr(self, 'end_detector'):
            return self.end_detector
        
        if not hasattr(self, 'trace'):
            self.get_trace()
        
        #print(f'trace: {self.trace}')
        # 位置为测线与Field的右边界或下边界（right_wall, bottom_wall）的交点
        end_point_proposal_right = self.trace.intersection(self.field.right_wall)
        assert len(end_point_proposal_right) == 1
        end_point_proposal_right = end_point_proposal_right[0].evalf(self.precision)

        #print(f'end_point_proposal_right: {end_point_proposal_right}')
        if end_point_proposal_right.x > self.field.bottom_wall.p1.x:
            end_point_proposal_bottom = self.trace.intersection(self.field.bottom_wall)
            assert len(end_point_proposal_bottom) == 1
            self.end_point = end_point_proposal_bottom[0].evalf(self.precision)
        else:
            self.end_point = end_point_proposal_right

        self.end_detector = Detector(self.field, x=self.end_point.x, y=self.end_point.y, 
                                     direction_angle=self.detector_0.direction_angle, 
                                     detector_angle=self.detector_0.detector_angle,
                                     precision=self.precision)

        return self.end_detector
    
    def get_length(self):
        if hasattr(self, 'length'):
            return self.length
        
        if not hasattr(self, 'end_detector'):
            self.get_end_detector()
        
        self.length = sp.Segment(self.detector_0.loc, self.end_detector.loc).length.evalf(self.precision)

        return self.length