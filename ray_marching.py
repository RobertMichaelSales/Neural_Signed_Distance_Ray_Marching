import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# https://www.cl.cam.ac.uk/teaching/1819/FGraphics/1.%20Ray%20Marching%20and%20Signed%20Distance%20Fields.pdf

# https://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf

#==============================================================================

def Normalise(vector):
    
    vector = vector / np.linalg.norm(vector)
    
    return vector

##

#==============================================================================

def Transform(vector,original_centre,original_radius):
    
    return vector

##

#==============================================================================

class Viewer():
    
    def __init__(self,camera_origin,screen_normal,fov,resolution):
        
        self.camera_origin = camera_origin
        self.screen_normal = Normalise(screen_normal)
        self.fov = fov
        self.h_res = resolution[0]
        self.v_res = resolution[1]
        
    ##
        
    def GetPixelCentres(self):
    
        up_vector = np.array([0.0,1.0,0.0])
        
        lr_vector = ((Normalise(np.cross(self.screen_normal,up_vector)) * np.linalg.norm(self.screen_normal) * np.tan(self.fov/2) * 2.0) / self.v_res)
        
        tb_vector = ((Normalise(np.cross(self.screen_normal,lr_vector)) * np.linalg.norm(self.screen_normal) * np.tan(self.fov/2) * 2.0) / self.v_res)
        
        screen_points = np.stack([np.fromfunction(lambda i,j: self.camera_origin[k] + self.screen_normal[k] + ((i-(self.h_res/2))*lr_vector[k]) + ((j-(self.v_res/2))*tb_vector[k]), shape=(self.h_res+1,self.v_res+1)) for k in [0,1,2]],axis=-1)
                
        self.pixel_centres = np.multiply(0.5,(screen_points[:-1,:-1] + screen_points[ 1:, 1:]))
                
    ## 
    
##

#==============================================================================
# Line-sphere intersection from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

class Ray():
    
    def __init__(self,camera_origin,pixel_centre,sdf):
        
        self.camera_origin = camera_origin
        self.pixel_centre = pixel_centre
        self.sdf = sdf
        self.bounding_sphere_centre = sdf.original_centre
        self.bounding_sphere_radius = sdf.original_radius
        
        self.ray_direction = Normalise(pixel_centre - camera_origin)

    ##
    
    def Intersect(self):
        
        # Check whether ray will ever intersect bounding sphere of SDF
        
        c = self.bounding_sphere_centre
        r = self.bounding_sphere_radius
        o = self.camera_origin
        u = self.ray_direction
        
        discriminant = np.square(np.dot(u,(o - c))) - (np.square(np.linalg.norm(o - c)) - np.square(r))
        
        if (discriminant >= 0.0):
            ray_parameter_max = (-1*(np.dot(u,(o - c)))) + np.sqrt(discriminant)
            ray_parameter_min = (-1*(np.dot(u,(o - c)))) - np.sqrt(discriminant)
            return True,ray_parameter_min,ray_parameter_max
        else:
            return False,0.0,0.0
        ##
        
    ##
    
    def March(self):
        
        ray_intersects,ray_parameter_min,ray_parameter_max = self.Intersect()
         
        if ray_intersects:
            
            ray_parameter = ray_parameter_min
            
            ray_in_bounds = True
            
            ray_is_active = True
        
            sdf_tolerance = 0.01
            
            num_ray_steps = 0
        
            while(ray_in_bounds and ray_is_active and (num_ray_steps < 100)):
                
                ray_position = self.camera_origin + (self.ray_direction * (ray_parameter))
                
                value_at_ray = self.sdf.GetSDF(position=ray_position)
                                
                # Check for hit
                if (np.abs(value_at_ray) <= sdf_tolerance):
                    ray_in_bounds = True
                    ray_is_active = False
                    ray_hit = 1
                    continue
                else:
                    ray_in_bounds = True
                    ray_is_active = True
                    ray_hit = 0
                    pass
                ##
                
                # Check in bounds
                if (ray_parameter_min <= ray_parameter <= ray_parameter_max):
                    ray_in_bounds = True
                    ray_is_active = True
                    ray_hit = 0
                    pass
                else:
                    ray_in_bounds = False
                    ray_is_active = True
                    ray_hit = 0
                    continue
                ##
                
                ray_parameter = ray_parameter + (-1.0 * value_at_ray)
                num_ray_steps = num_ray_steps + 1
                
            ##
            
            ray_depth = np.linalg.norm(ray_position - self.camera_origin)
                    
        ##
        
        else:
            
            ray_hit = 0
            ray_depth = 0.0
            
            num_ray_steps = 0
            value_at_ray = 0.0
            ray_position = self.camera_origin
            
        ##       
        
        # return ray_hit,ray_depth
        return ray_hit,ray_depth,ray_intersects,num_ray_steps,value_at_ray,ray_position
        
    ##
    
##

#==============================================================================
# SDF From https://iquilezles.org/articles/distfunctions/

class Sphere():
    
    def __init__(self,centre,radius):
        
        self.centre = centre
        
        self.radius = radius
        
    ##
    
    def GetSDF(self,position):
    
        value = np.linalg.norm(position - self.centre) - self.radius
        
        return value
    
    ##
    
##

#==============================================================================
# SDF From https://iquilezles.org/articles/distfunctions/

class Cuboid():
    
    def __init__(self,centre,corner):
        
        self.centre = centre
        
        self.corner = corner
        
    ##
    
    def GetSDF(self,position):
    
        value = np.linalg.norm(np.maximum((np.abs(position - self.centre) - self.corner),np.array([0.0,0.0,0.0])))
                
        return value
    
    ##
    
##

#==============================================================================

class SDF():
    
    def __init__(self,architecture_path,parameters_path):
        
        from network_decoder_copy import DecodeParameters,DecodeArchitecture,AssignParameters
        from network_model_copy import ConstructNetworkBASIC
        
        layer_dimensions,frequencies = DecodeArchitecture(architecture_path=architecture_path)
        
        self.sdf = ConstructNetworkBASIC(layer_dimensions=layer_dimensions,frequencies=frequencies,activation="relu")    

        parameters,self.original_centre,self.original_radius = DecodeParameters(network=self.sdf,parameters_path=parameters_path)

        AssignParameters(network=self.sdf,parameters=parameters)  

    ##
    
    def GetSDF(self,position):
                        
        value = float(self.sdf(tf.reshape(tf.constant(position),shape=(1,3))))
        
        return value
    
    ##
    
##

#==============================================================================

camera_origin = np.array([+1.0,+1.0,+3.0])
    
screen_normal = Normalise(-1 * camera_origin)

fov = np.pi/4

resolution = (120*4,100*4)

viewer = Viewer(camera_origin=camera_origin,screen_normal=screen_normal,fov=fov,resolution=resolution)

viewer.GetPixelCentres()
    
architecture_path = "/home/rms221/Documents/Compressive_Neural_Signed_Distances/AuxFiles/outputs/squashsdf_savesdf/architecture.bin"
parameters_path = "/home/rms221/Documents/Compressive_Neural_Signed_Distances/AuxFiles/outputs/squashsdf_savesdf/parameters.bin"
sdf = SDF(architecture_path=architecture_path,parameters_path=parameters_path)

sdf.original_centre = np.array([0.0,0.0,0.0])
sdf.original_radius = np.array([1.0])

colour = np.zeros(shape=resolution)

ray_hits = np.zeros(shape=resolution)
ray_depths = np.zeros(shape=resolution)
ray_intersectss = np.zeros(shape=resolution)
num_ray_stepss = np.zeros(shape=resolution)
value_at_rays = np.zeros(shape=resolution)
ray_positions = np.zeros(shape=(resolution + (3,)))

for i in range(resolution[0]):
    
    for j in range(resolution[1]):
        
        ray = Ray(camera_origin=viewer.camera_origin,pixel_centre=viewer.pixel_centres[i,j],sdf=sdf)
        
        print(i,j)
                    
        # ray_hit,ray_depth = ray.March()
        ray_hit,ray_depth,ray_intersects,num_ray_steps,value_at_ray,ray_position = ray.March()
        
        colour[i,j] = ray_hit * ray_depth
        
        ray_hits[i,j] = ray_hit
        ray_depths[i,j] = ray_depth
        ray_intersectss[i,j] = ray_intersects
        num_ray_stepss[i,j] = num_ray_steps
        value_at_rays[i,j] = value_at_ray
        ray_positions[i,j,:] = ray_position
        
    ##
    
##

# binary_r: low = white, high = black

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot()
ax.imshow(colour.T,cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.title("colour")
plt.savefig("colour.png",bbox_inches="tight",dpi=600)  
plt.show()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot()
ax.imshow(ray_hits.T,cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.title("ray_hits")
plt.savefig("ray_hits.png",bbox_inches="tight",dpi=600)  
plt.show()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot()
ax.imshow(ray_depths.T,cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.title("ray_depth")
plt.savefig("ray_depth.png",bbox_inches="tight",dpi=600)  
plt.show()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot()
ax.imshow(ray_intersectss.T,cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.title("ray_intersect")
plt.savefig("ray_intersect.png",bbox_inches="tight",dpi=600)  
plt.show()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot()
ax.imshow(num_ray_stepss.T,cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.title("num_ray_steps")
plt.savefig("num_ray_steps.png",bbox_inches="tight",dpi=600)  
plt.show()

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot()
ax.imshow(value_at_rays.T,cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.title("value_at_ray")
plt.savefig("value_at_ray.png",bbox_inches="tight",dpi=600)  
plt.show()

##

# ray_positions = np.concatenate([ray_positions.reshape((-1,3)),np.array([[0.,0.,0.]])])
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(xs=ray_positions[:,0],ys=ray_positions[:,1],zs=ray_positions[:,2], c='b', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('3D Scatter Plot')
# ax.axis("equal")
# plt.show()

#==============================================================================

# camera_origin = np.array([-3.0,-2.0,-3.0])
    
# screen_normal = Normalise(-1 * camera_origin)

# fov = np.pi/6

# resolution = (120,100)

# viewer = Viewer(camera_origin=camera_origin,screen_normal=screen_normal,fov=fov,resolution=resolution)

# viewer.GetPixelCentres()
    
# cuboid = Cuboid(centre=np.array([0.0,0.0,0.0]),corner=np.array([0.5,0.5,0.5]))

# colour = np.zeros(shape=resolution)

# for i in range(resolution[0]):
    
#     for j in range(resolution[1]):
        
#         ray = Ray(camera_origin=viewer.camera_origin,pixel_centre=viewer.pixel_centres[i,j])
                    
#         ray_hit,ray_depth = ray.March(sdf=cuboid.GetSDF)
        
#         colour[i,j] = ray_hit * ray_depth
                
#     ##
    
# ##

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot()
# ax.imshow(np.flipud(colour.T),cmap="Greys",interpolation='nearest')
# ax.axis("equal")
# plt.show()

# ##

#==============================================================================

# for view_angle in np.linspace(start=-np.pi,stop=+np.pi,num=50):

#     camera_origin = np.array([-3.0*np.sin(view_angle),-3.0,-3.0*np.cos(view_angle)])
    
#     screen_normal = Normalise(-1 * camera_origin)
    
#     fov = np.pi/6
    
#     resolution = (120,100)
    
#     viewer = Viewer(camera_origin=camera_origin,screen_normal=screen_normal,fov=fov,resolution=resolution)
    
#     viewer.GetPixelCentres()
        
#     cuboid = Cuboid(centre=np.array([0.0,0.0,0.0]),corner=np.array([0.5,0.5,0.5]))
    
#     colour = np.zeros(shape=resolution)
    
#     for i in range(resolution[0]):
        
#         for j in range(resolution[1]):
            
#             ray = Ray(camera_origin=viewer.camera_origin,pixel_centre=viewer.pixel_centres[i,j])
                        
#             ray_hit,ray_depth = ray.March(sdf=cuboid.GetSDF)
            
#             colour[i,j] = ray_hit * ray_depth
                    
#         ##
        
#     ##
 
    
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot()
#     ax.imshow(np.flipud(colour.T),cmap="Greys",interpolation='nearest')
#     ax.axis("equal")
#     plt.show()
    
# ##

#==============================================================================