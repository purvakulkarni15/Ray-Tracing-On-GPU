# Ray-Tracing-On-GPU

### Libraries
<pre>
1. OpenCV - For Image Generation
</pre>

### Installation

#### Windows

<pre>
1. Download latest compatible version of opencv
2. Extract and copy the opencv dlls into the program folder or in the windows/system32 directory.
3. Create a new Visual Studio project.
4. Copy and paste the path to the opencv include folder in the project properties->vc++ directories->Include.
5. Copy and paste the path to the opencv library folder in the project properties->vc++ directories->Libraries.
6. Add the necessary opencv libraries to the properties->linker->Input
7. Last but not the least, add the source files to the project...And the project is ready to run!
</pre>

### What is Ray Tracing

![Ray Tracing](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Ray%20Tracing.png)

<pre>
Ray Tracing is a method of generating photo realistic images of the 3D scenes by tracing the path of light through each pixel in an image 
plane. Aim is to find the intersection of the rays with a scene consisting of a set of geometric primitives like polygons, spheres, cones 
etc.
</pre>

### Construction of Rays

<pre>
Ray Equation: P = P0 + t(D)
P0  : Focal Pt of camera (0, 0, -3.67).
D   : Direction vector
D = sensor_grid[i][j] – P0
The sensor grid is of the size 640 X 480 (Resolution of the Image generated).
</pre>

### Ray Tracing of Standard Shapes and Triangular Mesh

<pre>
1. This project has a menu driven display, which lets the user choose from the standard shapes (Plane, Sphere, Traingle, Box) and triangular 
mesh (OBJ File).
2. The user can choose any combination of the above objects to create the scene for Ray Tracing.
</pre>

### Algorithms

#### Ray-Plane Intersection

![Ray-Plane Intersection](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Ray-Plane%20Intersection.png)

#### Ray-Triangle Intersection

![Ray-Triangle Intersection](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Ray-Triangle%20Intersection.png)

#### Ray-Sphere Intersection

![Ray-Sphere Intersection](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Ray-Sphere%20Intersection.png)

#### Ray-Box Intersection

![Ray-Box Intersection](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Ray-Box%20Intersection.png)

### Accelerating Ray Tracing

#### Uniform Grid (Accelerating Data Structure)
 In a uniform grid the scene is uniformly divided into voxels and those voxels containing triangles or part of a triangle obtain a 
 reference to this triangle. 
 
 ##### Algorithm for Grid Construction
<pre>
1. Calculate the bounding cells (b1, b2) of Triangle Ti 
2. Test triangle-box intersection: Ti with every cell Cj 2 (b1, b2)
3. If triangle-box intersection returned true, add a reference of Ti to Cj. 
</pre>

##### Algorithm for Grid Traversal

![Uniform Grid Traversal](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Uniform%20Grid.png)

Fast Voxel Traversal Algorithm proposed by John Amanatides is implemented
<pre>
1. Initialization 7.2.1.1. Find the voxel coordinates of ray-Grid intersection point.  

   voxel_coordinate.x = (world_coordinate.x – scene.minPoint.x)/voxel_size.x 
   voxel_coordinate.y = (world_coordinate.y – scene.minPoint.x)/ voxel_size.y 
   voxel_coordinate.z = (world_coordinate.z – scene.minPoint.x)/ voxel_size.z  
   
2. Find tMaxX, tMaxY and tMaxZ which denote the initial positions at which the ray crosses the voxel boundaries.  

3. Find tDeltaX, tDeltaY and tDeltaZ which denote how far along the ray must be moved to equal the corresponding voxel lengths in X, Y 
   and Z directions. 

4. Initialize stepX, stepY and stepZ as either 1 or -1 depending upon the ray direction for each of its components. 

5. Incremental Traversal 

bool c1 = bool (tMax.x < tMax.y ) ; 
bool c2 = bool (tMax.x < tMax.z ) ; 
bool c3 = bool (tMax.y < tMax.z ) ; 

if (c1 && c2 ) 
{ 
  voxel.x += stepX; if (voxel.x==outX) return; // out of voxel space 
  tMax.x += tDelta.x; 
} 
else if ( ( ( c1 && ! c2 ) | | ( ! c1 && ! c3 ) ) ) 
{ 
  voxel.z += stepZ; if (voxel.z==outZ) return; // out of voxel space 
  tMax.z += tDelta.z; 
} 
else if ( ! c1 && c3 ) 
{ 
  voxel.y += stepY; if (voxel.y==outY) return; // out of voxel space 
  tMax.y += tDelta.y; 
}         
test (voxel ) ;//check  if ( voxel . x , voxel . y , voxel . z ) contains data
</pre>

### Results

![Standford Dragon](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/RayTraced_Dragon.jpg)

![Room](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Room.jpg)

![Cornel Box](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Cornell_Box.jpg)

![Spheres](https://github.com/purvakulkarni15/Ray-Tracing-On-GPU/blob/master/Spheres.jpg)
