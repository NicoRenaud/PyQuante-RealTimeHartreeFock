from PyQuante.Constants import bohr2ang,ang2bohr
from PyQuante import Molecule
import numpy as np
import sys
import ase.io.cube
import os
import re

"""
 Code for reading/writing Gaussian Cube files
 
 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""

# This code doesn't work!!! By which I mean that it works even
# less than the normal code in PyQuante ;-).
#
# Still in the process of being debugged.


####################################################
#           Get the size of the box
####################################################
def get_bbox(atoms,**kwargs):
    dbuff = kwargs.get('dbuff',2.5)
    big = kwargs.get('big',10000)
    xmin = ymin = zmin = big
    xmax = ymax = zmax = -big

    # get the limit of the atoms
    for atom in atoms:
        x,y,z = atom.pos()
        xmin = min(xmin,x)
        ymin = min(ymin,y)
        zmin = min(zmin,z)
        xmax = max(xmax,x)
        ymax = max(ymax,y)
        zmax = max(zmax,z)

    # add the buffers
    xmin -= dbuff
    ymin -= dbuff
    zmin -= dbuff
    xmax += dbuff
    ymax += dbuff
    zmax += dbuff

    # round to 0.5 Ang
    xmin = np.sign(xmin)*np.ceil(2*np.abs(xmin))/2
    ymin = np.sign(ymin)*np.ceil(2*np.abs(ymin))/2
    zmin = np.sign(zmin)*np.ceil(2*np.abs(zmin))/2
    xmax = np.sign(xmax)*np.ceil(2*np.abs(xmax))/2
    ymax = np.sign(ymax)*np.ceil(2*np.abs(ymax))/2
    zmax = np.sign(zmax)*np.ceil(2*np.abs(zmax))/2
    
    return (xmin,xmax),(ymin,ymax),(zmin,zmax)

####################################################
#          Mesh the orbitals
####################################################
def mesh_orb(file_name,atoms,bfs,orbs,index):
        
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = get_bbox(atoms)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    ppb = 2.0 # Points per bohr 
    spacing = 1.0/ppb
    nx,ny,nz = int(dx*ppb)+1,int(dy*ppb)+1,int(dz*ppb)+1

    
    print("\t\tWriting Gaussian Cube file {}".format(file_name))
    f = open(file_name,'w')

    f.write("CUBE FILE\n")
    f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (len(atoms),xmin,ymin,zmin))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nx,spacing,0,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (ny,0,spacing,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nz,0,0,spacing))

    # The second record here is the nuclear charge, which differs from the
    #  atomic number when a ppot is used. Since I don't have that info, I'll
    #  just echo the atno
    for atom in atoms:
        atno = atom.atno
        x,y,z = atom.pos()
        f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (atno,atno,x,y,z))

    nbf = len(bfs)
    f.write(" ")

    for i in xrange(nx):
        xg = xmin + i*spacing

        for j in xrange(ny):
            yg = ymin + j*spacing

            for k in xrange(nz):
                zg = zmin + k*spacing

                amp = 0

                for ibf in xrange(nbf):
                    amp += bfs[ibf].amp(xg,yg,zg)*orbs[ibf,index]

                if abs(amp) < 1e-12: 
                    amp = 0
                f.write(" %11.5e" % amp.real)
                if k % 6 == 5: 
                    f.write("\n")
            f.write("\n")
    f.close()
    xyz_min = np.array([xmin,ymin,zmin])
    np_pts = np.array([nx,ny,nz])
    return xyz_min,np_pts,spacing

####################################################
#          Mesh a general pop of the bfs
####################################################
def mesh_dens(file_name,atoms,bfs,density):
        
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = get_bbox(atoms)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    ppb = 2.0 # Points per bohr 
    spacing = 1.0/ppb
    nx,ny,nz = int(dx*ppb)+1,int(dy*ppb)+1,int(dz*ppb)+1

    
    print("\t\tWriting Gaussian Cube file {}".format(file_name))
    f = open(file_name,'w')

    f.write("CUBE FILE\n")
    f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (len(atoms),xmin,ymin,zmin))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nx,spacing,0,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (ny,0,spacing,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nz,0,0,spacing))

    # The second record here is the nuclear charge, which differs from the
    #  atomic number when a ppot is used. Since I don't have that info, I'll
    #  just echo the atno
    for atom in atoms:
        atno = atom.atno
        x,y,z = atom.pos()
        f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (atno,atno,x,y,z))

    nbf = len(bfs)
    f.write(" ")

    for i in xrange(nx):
        xg = xmin + i*spacing

        for j in xrange(ny):
            yg = ymin + j*spacing

            for k in xrange(nz):
                zg = zmin + k*spacing

                amp = 0

                for ibf in xrange(nbf):
                    amp += (bfs[ibf].amp(xg,yg,zg)*density[ibf])**2

                if abs(amp) < 1e-12: 
                    amp = 0
                f.write(" %11.5e" % amp.real)
                if k % 6 == 5: 
                    f.write("\n")
            f.write("\n")
    f.close()
    xyz_min = np.array([xmin,ymin,zmin])
    np_pts = np.array([nx,ny,nz])
    return xyz_min,np_pts,spacing


####################################################
#          Mesh a general pop of the bfs
####################################################
def mesh_dens_uhf(file_name,atoms,bfs,density_a,density_b,type_dens='total'):
        
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = get_bbox(atoms)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    ppb = 2.0 # Points per bohr 
    spacing = 1.0/ppb
    nx,ny,nz = int(dx*ppb)+1,int(dy*ppb)+1,int(dz*ppb)+1

    
    print("\t\tWriting Gaussian Cube file {}".format(file_name))
    f = open(file_name,'w')

    f.write("CUBE FILE\n")
    f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (len(atoms),xmin,ymin,zmin))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nx,spacing,0,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (ny,0,spacing,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nz,0,0,spacing))

    # The second record here is the nuclear charge, which differs from the
    #  atomic number when a ppot is used. Since I don't have that info, I'll
    #  just echo the atno
    for atom in atoms:
        atno = atom.atno
        x,y,z = atom.pos()
        f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (atno,atno,x,y,z))

    nbf = len(bfs)
    f.write(" ")

    for i in xrange(nx):
        xg = xmin + i*spacing

        for j in xrange(ny):
            yg = ymin + j*spacing

            for k in xrange(nz):
                zg = zmin + k*spacing

                amp = 0

                for ibf in xrange(nbf):
                    if type_dens == 'total':
                        amp += (bfs[ibf].amp(xg,yg,zg)*density_a[ibf])**2 + (bfs[ibf].amp(xg,yg,zg)*density_b[ibf])**2
                    if type_dens == 'spin':
                        amp += (bfs[ibf].amp(xg,yg,zg)*density_a[ibf])**2 - (bfs[ibf].amp(xg,yg,zg)*density_b[ibf])**2
                if abs(amp) < 1e-12: 
                    amp = 0
                f.write(" %11.5e" % amp.real)
                if k % 6 == 5: 
                    f.write("\n")
            f.write("\n")
    f.close()
    xyz_min = np.array([xmin,ymin,zmin])
    np_pts = np.array([nx,ny,nz])
    return xyz_min,np_pts,spacing



####################################################
#          Create the volumetric
#           data for blender
####################################################
def cube2blender(fname):

    #print("Reading cube file {}".format(fname))
    data, atoms = ase.io.cube.read_cube_data(fname)

    # Here, I want the electron density, not the wave function
    data = data**2

    # If data is too large, just reduce it by striding with steps >1
    sx, sy, sz = 1, 1, 1
    data = data[::sx,::sy,::sz]

    # Note the reversed order!!
    nz, ny, nx = data.shape
    nframes = 1
    header = np.array([nx,ny,nz,nframes])

    #open and write to file
    vdata = data.flatten() / np.max(data)
    vfname = os.path.splitext(fname)[0] + '.bvox'
    vfile = open(vfname,'wb')
    print("\t\tWriting Blender voxel file {}".format(vfname))
    header.astype('<i4').tofile(vfile)
    vdata.astype('<f4').tofile(vfile)

    return vfname

####################################################
#          Create the script
#            for blender
####################################################
def create_blender_script(xyz_min,nb_pts,spacing,pdbfile,bvoxfiles,path_to_files):

    # center/size of the 
    xyz_max = xyz_min+(nb_pts-1)*spacing
    
    center = bohr2ang*0.5*(xyz_max+xyz_min)/10
    size = bohr2ang*(xyz_max-xyz_min)/10/2

    f = open('blender_volumetrix.py','w')
    f.write('import blmol\n')
    f.write('import bpy\n')
    f.write('import os\n\n')
    f.write("#go in the directories where the files are\n")
    f.write("#you may want to change that if the rendering\n")
    f.write("#and the calculations are not done on the same machine\n")
    f.write("os.chdir('%s')\n" %(path_to_files))
    f.write("\n")
    f.write("############################\n")
    f.write("# Tags to initialize\n")
    f.write("# or render the orbitals\n")
    f.write("############################\n")
    f.write("# keep _init_scene_ set to 1 and run the script\n")
    f.write("# Tweek the scene to your likings\n")
    f.write("# set _init_scene_=0 and _render_scene_=1 \n")
    f.write("# and re-run the script to render the MOs \n")
    f.write("############################\n")
    f.write('_init_scene_=1\n')
    f.write('_render_scene_=0\n')
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# function to create the material\n")
    f.write("# we want for the orbitals\n")
    f.write("############################\n")
    f.write("def makeMaterial(name_mat,name_text,bvoxfile):\n")
    f.write("\n")
    f.write("    # create the material\n")
    f.write("    mat = bpy.data.materials.new(name_mat)\n")
    f.write("    mat.type = 'VOLUME'\n")
    f.write("    mat.volume.density = 0.0\n")
    f.write("    mat.volume.emission = 10.0\n")
    f.write("\n")
    f.write("    # create the texture\n")
    f.write("    text = bpy.data.textures.new(name_text,type='VOXEL_DATA')\n")
    f.write("    text.voxel_data.file_format = 'BLENDER_VOXEL'\n")
    f.write("    text.voxel_data.filepath = bvoxfile\n")
    f.write("    text.voxel_data.use_still_frame = True\n")
    f.write("    text.voxel_data.still_frame = 1\n")
    f.write("    text.voxel_data.intensity = 10\n")
    f.write("    \n")
    f.write("    # use a color ramp\n")
    f.write("    text.use_color_ramp = True\n")
    f.write("    text.color_ramp.elements.new(0.5)\n")
    f.write("    text.color_ramp.elements[0].color = (0.0,0.0,0.0,0.0)\n")
    f.write("    text.color_ramp.elements[1].color = (0.0,0.30,0.50,1.0)\n")
    f.write("    text.color_ramp.elements[2].color = (1.0,1.0,1.0,0.0)\n")
    f.write("    \n")
    f.write("    # add the texture to the amt\n")
    f.write("    mtex = mat.texture_slots.add()\n")
    f.write("    mtex.texture = text\n")
    f.write("    mtex.texture_coords = 'ORCO'\n")
    f.write("    mtex.use_map_density = True\n")
    f.write("    mtex.use_map_emission = True\n")
    f.write("    mtex.use_from_dupli = False\n")
    f.write("    mtex.use_map_to_bounds = False\n")
    f.write("    mtex.use_rgb_to_intensity = False\n")
    f.write("    \n")
    f.write("    # return the mat\n")
    f.write("    return mat\n")
    f.write("############################\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# function to assigne a material\n")
    f.write("# to an oject\n")
    f.write("############################\n")
    f.write("def setMaterial(ob, mat):\n")
    f.write("    me = ob.data\n")
    f.write("    me.materials.append(mat)\n")
    f.write("############################\n")
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# Function to initialize \n")
    f.write("# the Blender scene \n")
    f.write("############################\n")
    f.write("def init_scene():\n")
    f.write("\n")
    f.write("   # change the render setting\n")
    f.write("   bpy.context.scene.render.resolution_x = 2000\n")
    f.write("   bpy.context.scene.render.resolution_y = 2000\n")
    f.write("\n")
    f.write("   # change the horizon color\n")
    f.write("   bpy.context.scene.world.horizon_color = (0,0,0)\n")
    f.write("\n")
    f.write("   # change the camera position\n")
    f.write("   cam = bpy.data.objects['Camera']\n")
    f.write("   cam.location.x = %f\n" %center[0])
    f.write("   cam.location.y = %f\n" %center[1])
    f.write("   cam.location.z = %f\n" %(center[2]+2.))
    f.write("   cam.rotation_euler[0] = 0\n")
    f.write("   cam.rotation_euler[1] = 0\n")
    f.write("   cam.rotation_euler[2] = 0\n")
    f.write("\n")
    f.write("   #Create the molecule\n")
    f.write('   m = blmol.Molecule()\n')
    f.write("   m.read_pdb('%s')\n" %(pdbfile))
    f.write('   m.draw_bonds()\n')
    f.write("\n")
    f.write("   #Create the material\n")
    f.write("   momat = makeMaterial('momat','motext','%s')\n\n" %(bvoxfiles[0]))
    f.write("   # create the cube that we use to display the volume\n")    
    f.write('   bpy.ops.mesh.primitive_cube_add(location=(%f,%f,%f))\n' %(center[0],center[1],center[2]))
    f.write('   bpy.ops.transform.rotate(value=%f,axis=(0.0,1.,0.0))\n' %(np.pi/2))
    f.write('   bpy.ops.transform.resize(value=(%f,%f,%f))\n' %(size[0],size[1],size[2]))
    f.write("\n")
    f.write("   # Assign the material to the cube\n") 
    f.write('   setMaterial(bpy.context.object, momat)\n')
    f.write("############################\n")
    f.write("# Function to render all \n")
    f.write("# the bvoxfiles \n")
    f.write("############################\n")
    f.write("def render_files():\n")
    for iF in range(len(bvoxfiles)):
        image_name = bvoxfiles[iF][:-4]+'jpg'
        f.write("  bpy.data.textures['motext'].voxel_data.filepath = '%s' \n" %(bvoxfiles[iF]))
        f.write("  bpy.data.scenes['Scene'].render.filepath = '%s'\n" %image_name)
        f.write("  bpy.ops.render.render( write_still=True )\n\n")
    f.write("############################\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# switch to init or render the scene\n")
    f.write("############################\n")
    f.write("if _init_scene_ == 1:\n")
    f.write("   init_scene()\n")
    f.write("elif _render_scene_==1:\n")
    f.write("   render_files()\n")
    f.write("############################\n")
    f.close()

#####################################################
##  create a pdb file from the xyz
#####################################################
def create_pdb(pdb_file,xyz_file,units):

    # create the pdb file if it does not exists
    if not os.path.isfile(pdb_file):

        # if it was provided in bohr 
        # change to angstom
        if units == 'bohr':

            # read the file
            f = open(xyz_file,'r')
            data = f.readlines()
            f.close()

            #write a xyz file in angstrom
            name_mol = re.split(r'\.|/',xyz_file)[-2]
            fname = name_mol+'_angs.xyz'
            f = open(fname,'w')
            f.write('%s\n' %data[0])
            for i in range(2,len(data)):
                l = data[i].split()
                if len(l)>0:
                    x,y,z = bohr2ang*float(l[1]),bohr2ang*float(l[2]),bohr2ang*float(l[3])
                    f.write('%s %f %f %f\n' %(l[0],x,y,z))
            f.close()

            # convert to pdb
            os.system('obabel -ixyz %s -opdb -O %s' %(fname,pdb_file))

