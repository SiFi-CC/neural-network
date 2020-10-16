# Author: Awal Awal
# Date: May 2020
# Email: awal.nova@gmail.com

class SiFiCC_Module:
    '''Represents a single module (scatterer or absorber) within the SiFi-CC
    '''
    
    def __init__(self, thickness_x, thickness_y, thickness_z, position, orientation=0):
        self.thickness_x = thickness_x
        self.thickness_y = thickness_y
        self.thickness_z = thickness_z
        self.position = position
        self.orientation = orientation
        
        self.start_x = self.position.x - self.thickness_x/2
        self.end_x = self.position.x + self.thickness_x/2
        self.start_y = self.position.y - self.thickness_y/2
        self.end_y = self.position.y + self.thickness_y/2
        self.start_z = self.position.z - self.thickness_z/2
        self.end_z = self.position.z + self.thickness_z/2
        
    def is_point_inside_x(self, point):
        '''Checks if `point` is within the module based on the x-axis only.
        '''
        if self.start_x < point.x < self.end_x:
            return True
        else:
            return False
        
    def is_any_point_inside_x(self, l_points):
        '''Checks if any point in `l_points` is within the module based on the x-axis only
        '''
        for point in l_points:
            if self.is_point_inside_x(point):
                return True
        else:
            return False