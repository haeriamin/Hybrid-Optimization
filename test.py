# Copyright (C) 2020 Amin Haeri [ahaeri92@gmail.com]
# min force on load: -69.824

# For non-MVS users:
import sys
# sys.path.append('../../../../../../CM Labs/Vortex Studio 2020.4/bin')
sys.path.append('C:/CM Labs/Vortex Studio 2020.4/bin')


import VxSim
from VxSim import *
from vxatp import *

import shutil
import time
import os
import math
import csv
# import numpy as np

import partio


# ------------------------------------------------------------------------------
# definitions
# ------------------------------------------------------------------------------
def get_extension_from_object(object, extensionName):
    extensions = object.getExtensions()
    for extension in extensions:
        if extension.getName() == extensionName:
            return extension


def get_mechanism_from_scene(scene_to_look_into, mechanism_to_look_for):
    for mechanism in scene_to_look_into.getMechanisms():
        if mechanism_to_look_for == mechanism.getName():
            return mechanism


def get_part_from_assembly(assembly_to_look_into, part_to_look_for):
    for part in assembly_to_look_into.getParts():
        if part_to_look_for == part.getName():
            return part


def get_constraint_from_assembly(assembly_to_look_into, constraint_to_look_for):
    for constraint in assembly_to_look_into.getConstraints():
        if constraint_to_look_for == constraint.getName():
            return constraint


def get_assembly_from_mechanism(mechanism_to_look_into, assembly_to_look_for):
    for assembly in mechanism_to_look_into.getAssemblies():
        if assembly_to_look_for == assembly.getName():
            return assembly


def natural_freq_Hz(densityVal, radii, k, n):
    tao = math.sqrt(densityVal*4/3*math.pi*(min(radii))**3/k)*2*math.pi
    freq = int(math.ceil((2*n)/tao)/100)*100
    if freq < 5000:
        freq = 5000
    return freq


def damping_coef(densityVal, radii, k, u):
    cc = math.sqrt(densityVal*4/3*math.pi*(min(radii))**3*k)*2
    c = math.ceil(cc*u)
    if u is 0.0:
        c = 0.0
    return c


def particle_mean_velocity(grainVel, numParticles):  # velocity of particles
    maxNormVel = None
    aveNormVel = 0.0
    for grain in range(numParticles):
        vel = np.array(grainVel[grain])
        normVel = np.linalg.norm(vel)
        aveNormVel = aveNormVel+normVel
        if maxNormVel is None:
            maxNormVel = normVel
        else:
            if normVel > maxNormVel:
                maxNormVel = normVel
    aveNormVel = aveNormVel/numParticles
    return [aveNormVel, maxNormVel]


# function to create the part definition
class ExcavDefinition:
    def __init__(self):
        self.height = 0.305  # m (was .08)
        self.length = 0.006
        self.width = 0.457
        self.slope_angle = -3.8  # deg
        self.forward_velocity = 0.04  # m/s
        self.vertical_velocity = 0.025  # m/s
        self.forward_time = 19.38  # sec
        self.vertical_time = 2  # sec
        self.depth = 0.05  # m
        self.mass = 1.743  # kg
        self.material = 'Wheel'
        self.groupId = 0


class BinDefinition:
    def __init__(self):
        scale = 0.5
        self.height = .45  # To reduce run time
        self.length = 1.2  #1.5 # 2. 
        self.width = .8  # 1.
        self.mass = 2.
        self.material = 'Ground'
        self.groupId = 0


# ------------------------------------------------------------------------------
# select type of simulation
# ------------------------------------------------------------------------------
runType = 'fullRun'


# ------------------------------------------------------------------------------
# create output directory
# ------------------------------------------------------------------------------
VortexOutputPath = "./input/"
CSVoutputPath = "./test/"


# ------------------------------------------------------------------------------
# choose how to run
# ------------------------------------------------------------------------------
binDef = BinDefinition()
excavDef = ExcavDefinition()

# what to run
mode = 'hybrid'
moving_excav = False
estop = 20000000000  # steps
excavXstart = -0.95*binDef.length/2
soil_height = 0.230 #0.145  # [m]
forward_velocity = excavDef.forward_velocity  # [m/s]
vertical_velocity = excavDef.vertical_velocity  # [m/s]
maxT = excavDef.forward_time + 2*excavDef.vertical_time # sim time
g = 9.81  # [m/s2]

# Soil properties & Simulation presets
# 5cm:
# friction_coef = 6.83741285e-01
# friction_coef_soil_tool = 3.35455728e-01
# radius = 1.10511245e-02
# surcharge_factor = 2.11047748e-01
# iteration_number = 19
# 2cm:
# friction_coef = 7.04127367e-01
# friction_coef_soil_tool = 3.35380587e-01
# radius = 1.09857695e-02
# surcharge_factor = 3.47156506e-01
# iteration_number = 19
# Both:
friction_coef = 7.11653299e-01
friction_coef_soil_tool = 3.97859022e-01
radius = 1.19631175e-02
surcharge_factor = 2.47373440e-01
iteration_number = 19

CF = 0.0  # Collision error compensation factor
density_grain = 2580.0  # [kg/m^3]
adhesion = 0.0  # [Pa]
c_coef = 2500.0
E = 150000.  # Young's modulus [Pa]
k = math.pi/2. * E * radius  # Stiffness for monodisperse material => 1414 Pa
c = 2065887.0  # = damping_coef(densityVal, radius, k, c_coef)
density_bulk = 1730  # [kg/m3]
porosity = 1 - (density_bulk/2.583/1000)  # 0.330
void_ratio = porosity / (1 - porosity)  #  0.493
volume_max = 4.0/3.0*math.pi*(1.1*radius)**3 

frame_rate = 60
vidFrameRate = 60  # Always a factor of 60
if vidFrameRate < frame_rate:
    SimFrames = math.floor(frame_rate/vidFrameRate)
else:
    SimFrames = 1
k_digits = int(math.log10(k))+1
if c == 0.0:
    c_digits = 1
else:
    c_digits = int(math.log10(c))+1
presetName = str(adhesion)+'a_'+str(int(k/(10**(k_digits-1))))+'e'+str(k_digits-1)+'k_'+str(int(c/(10**(
    c_digits-1))))+'e'+str(c_digits-1)+'c_'+str(float(int(friction_coef*100))/100.0)+'f_'+str(int(density_grain))+'d'


# ------------------------------------------------------------------
# Run application
# ------------------------------------------------------------------
application = None
application = VxApplication()

serializer = ApplicationConfigSerializer()
serializer.load('./input/editor-no-gfx.vxc')
# serializer.load('./input/editor_500Hz.vxc')

# Extract the ApplicationConfig
config = serializer.getApplicationConfig()
config.apply(application)

# VxATPUtils.requestApplicationModeChangeAndWait(application, kModeEditing)

# Get the file manager to load content
fileManager = application.getSimulationFileManager()

# Set Simulation Frame Rate
application.setSimulationFrameRate(frame_rate)

# Set Gravity
# dynamics = application.getModule(0).getExtension()
# gravity = dynamics.getParameter('Gravity')
# gravity.setValue(VxVector3(0, 0, -g))

# Load the file, the object returned is the root object, in this case, a scene.
# The Reference should be kept to inspect the content during the simulation.
# The object loaded is already added to the application
# SceneName = 'testbed_Spontaneous1'
SceneName = 'testbed_Spontaneous1_motion'
scene = VxSim.SceneInterface(fileManager.loadObject(VortexOutputPath + '/' + SceneName + '.vxscene'))

if scene.valid():
    soil_materials_extension = get_extension_from_object(scene, 'Soil Materials')
    soil_materials = soil_materials_extension.getExtension()

    CartMechanism = get_mechanism_from_scene(scene, 'Cart')
    CartAssembly = get_assembly_from_mechanism(CartMechanism, 'Cart')

    ExcavPart = get_part_from_assembly(CartAssembly, 'excav')
    
    load_extension = get_constraint_from_assembly(CartAssembly, 'Load')
    load = PrismaticInterface(load_extension.getExtension())
    
    linear_extension = get_constraint_from_assembly(CartAssembly, 'Linear')
    linear = PrismaticInterface(linear_extension.getExtension())

    PoolMechanism = get_mechanism_from_scene(scene, 'Pool')
    PoolAssembly = get_assembly_from_mechanism(PoolMechanism, 'Pool')

    soil_dynamics_extension = get_extension_from_object(scene, 'Dynamics Soil Mesh')
    soil_dynamics = soil_dynamics_extension.getExtension()
else:
    print 'Error: scene not found'


# ------------------------------------------------------------------
# Setting
# ------------------------------------------------------------------

# TODO: Excav position should be set here
# Cart
# cartPos = VxSim.VxVector3(excavXstart, 0., binDef.height+excavDef.height/2+0.1)


# TODO: Soil's height should be set here
# Pool
# ? = VxSim.VxVector3(0., 0., soil_height)

# Soil
# General
soil_materials.getInput('Target Preset File Name').setValue(CSVoutputPath + presetName + '.vxr')
soil_materials.getInput('Advanced').setValue(True)

# Soil Properties
soil_materials.getInput('Soil Material Preset Parameters')[
    'Soil Properties']['FEE Tool Soil Friction Angle'].setValue(math.atan(friction_coef_soil_tool)*180/math.pi)
soil_materials.getInput('Soil Material Preset Parameters')[
    'Soil Properties']['Maximum Particle Volume'].setValue(volume_max)
soil_materials.getInput('Soil Material Preset Parameters')[
    'Soil Properties']['FEE Surcharge Contribution Factor'].setValue(surcharge_factor)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'Soil Properties']['Density'].setValue(density_grain)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'Soil Properties']['Minimum Void Ratio'].setValue(void_ratio)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'Soil Properties']['Maximum Void Ratio'].setValue(void_ratio)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'Soil Properties']['Internal Friction Coefficient Table'][0]['Void Ratio'].setValue(void_ratio)
soil_materials.getInput('Soil Material Preset Parameters')[
    'Soil Properties']['Internal Friction Coefficient Table'][0]['Strength Property'].setValue(friction_coef)

# PB Generator Properties
# General    
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Collision Error Compensation Factor'].setValue(CF)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Solver Version'].setValue('Latest Solver')
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Max Acceleration'].setValue(-1.0)
soil_materials.getInput('Soil Material Preset Parameters')[
    'PB Generator Properties']['Solver Iteration Count'].setValue(iteration_number)
# Particle Material
soil_materials.getInput('Soil Material Preset Parameters')[
    'PB Generator Properties']['Particle Material']['Friction Coefficient'].setValue(friction_coef)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle Material']['Adhesion'].setValue(adhesion)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle Material']['Stiffness Coefficient'].setValue(k)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle Material']['Damping Coefficient'].setValue(c)
# Particle/Particle Material
soil_materials.getInput('Soil Material Preset Parameters')[
    'PB Generator Properties']['Particle/Particle Material']['Friction Coefficient'].setValue(friction_coef)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle/Particle Material']['Adhesion'].setValue(adhesion)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle/Particle Material']['Stiffness Coefficient'].setValue(k)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle/Particle Material']['Damping Coefficient'].setValue(c)
# Particle/Zone Material
soil_materials.getInput('Soil Material Preset Parameters')[
    'PB Generator Properties']['Particle/Zone Material']['Friction Coefficient'].setValue(friction_coef)
# Particle/Tool Material
soil_materials.getInput('Soil Material Preset Parameters')[
    'PB Generator Properties']['Particle/Tool Material']['Friction Coefficient'].setValue(friction_coef_soil_tool)
# soil_materials.getInput('Soil Material Preset Parameters')[
#     'PB Generator Properties']['Particle/Tool Material']['Adhesion'].setValue(adhesion)

VxATPUtils.requestApplicationModeChangeAndWait(application, kModeSimulating)

# ------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------
t = 0
dt = application.getSimulationTimeStep()
excavDef = ExcavDefinition()
end = 0
step = -1
framecount = 0
check_state_update = SimFrames*4
hist_ips = int(math.ceil(frame_rate/check_state_update))
state = 'start'
numParticles = 0
grainPos = None
filename = 'hybrid' # + str(time.time())
dataPath = CSVoutputPath+filename
if os.path.exists(dataPath):
    shutil.rmtree(dataPath)
    os.makedirs(dataPath)
else:
    os.makedirs(dataPath)

TIC = time.time()


# Simulation Loop
# ------------------------------------------------------------------
while end is 0:

    # ----------------------------------------------------------
    # Extract Relevant data from application
    # ----------------------------------------------------------
    if ((step % SimFrames == 0.0) or (step % check_state_update == 0.0)) and (step != -1):
        particles_container = soil_dynamics.getOutput('Particles')
        grainDia = particles_container['Radii'].toVectorFloat()
        grainPos = particles_container['Positions'].toVectorVector3f()
        numParticles = len(grainDia)
        numParticles_min = min(len(previousGrainPos),len(grainPos))
        grainVel = []  # [X_vel, Y_vel, Z_vel] [m/s]
        for grain in range(numParticles_min):
            position = grainPos[grain]
            if step > -1:
                # difference between current and previous particle positions
                diff = [position[0] - previousGrainPos[grain][0],
                        position[1] - previousGrainPos[grain][1],
                        position[2] - previousGrainPos[grain][2]]
                # difference in positions divided by time step size
                velocity = [x/dt for x in diff]
            else:
                velocity = [0, 0, 0]
            grainVel.append(velocity)

        lift = load.outputLinearCoordinate.currentStateControlForce.value
        drawbarPull = linear.outputLinearCoordinate.currentStateControlForce.value

        excavPos = getTranslation(ExcavPart.outputWorldTransform.value)
        excavVel = ExcavPart.outputLinearVelocity.value


    # ------------------------------------------------------
    # Update State of Simulation
    # ------------------------------------------------------
    if (step % check_state_update == 0.0) and (step != -1):
        if (state is 'start'):
            load.inputLinearCoordinate.motor.desiredVelocity.value = 0.
            linear.inputLinearCoordinate.motor.desiredVelocity.value = 0.
            tic = time.time()
            state = 'settled'
            t_wait = 0

        if (state is 'settled'):
            if excavPos[2] > excavDef.height/2 + 1.1*soil_height:
                # Lower excav quickly until just above surface
                load.inputLinearCoordinate.motor.desiredVelocity.value = -1.0
            else:
                # Lower excav slowly into soil to avoid high-speed collision
                linear.inputLinearCoordinate.motor.desiredVelocity.value = vertical_velocity  # forward
                load.inputLinearCoordinate.motor.desiredVelocity.value = -vertical_velocity  # vertical
                state = 'lower'
                t_moving = 0.0

        if (state is 'lower'):  # Excav is within buffer of soil surface
            Height = excavDef.height/2*math.sin(math.pi/180*(excavDef.slope_angle+90)) - excavDef.depth + soil_height
            if excavPos[2] <= 1.02*Height:
                linear.inputLinearCoordinate.motor.desiredVelocity.value = 0.  # forward
                load.inputLinearCoordinate.motor.desiredVelocity.value = 0.  # vertical
                state = 'forward'
                moving_excav = True

        if (state is 'forward'):  # After excav lowering, excav motion can begin
            linear.inputLinearCoordinate.motor.desiredVelocity.value = forward_velocity  # forward
            load.inputLinearCoordinate.motor.desiredVelocity.value = 0.  # vertical
            # if t_moving > excavDef.vertical_time/2 + excavDef.forward_time:
            if excavPos[0] >= 0.39*binDef.length/2:
                state = 'finished'
            #     state = 'up'

        # if (state is 'up'):  # After excav lowering, excav motion can begin
        #     linear.inputLinearCoordinate.motor.desiredVelocity.value = vertical_velocity  # forward
        #     load.inputLinearCoordinate.motor.desiredVelocity.value = vertical_velocity  # vertical
        #     Height = excavDef.height/2*math.sin(math.pi/180*(excavDef.slope_angle+90)) - excavDef.depth + soil_height
        #     if excavPos[2] > 1.1*Height:
        #         state = 'finished'


    TOC = time.time()
    if step is -1:
        python_step_time = (TOC-TIC)
    else:
        python_step_time = (TOC-TIC)-(TOCC-TICC)


    # ------------------------------------------------------
    # Save Outputs
    # ------------------------------------------------------
    if (step % SimFrames == 0.0) and (step != -1):
        toc = time.time()

        # Houdini: Rigid body (plate)
        obj_file = open(dataPath+'/'+str(framecount)+'.obj', 'w+')
        dim = 3
        sa  = math.pi/180 * (excavDef.slope_angle+90)
        aRB = excavPos[0] - excavDef.length/2 + excavDef.height/2*math.cos(sa)
        aRT = excavPos[0] - excavDef.length/2 - excavDef.height/2*math.cos(sa)
        aFB = excavPos[0] + excavDef.length/2 + excavDef.height/2*math.cos(sa)
        aFT = excavPos[0] + excavDef.length/2 - excavDef.height/2*math.cos(sa)
        bL  = excavPos[1] + excavDef.width/2
        bR  = excavPos[1] - excavDef.width/2
        cB  = excavPos[2] - excavDef.height/2*math.sin(sa)
        cT  = excavPos[2] + excavDef.height/2*math.sin(sa)
        # Vertex
        # in x
        obj_file.write("v {}, {}, {}\n".format(aRB, bL, cB))
        obj_file.write("v {}, {}, {}\n".format(aRB, bR, cB))
        obj_file.write("v {}, {}, {}\n".format(aRT, bR, cT))
        obj_file.write("v {}, {}, {}\n".format(aRT, bL, cT))
        #    
        obj_file.write("v {}, {}, {}\n".format(aFB, bL, cB))
        obj_file.write("v {}, {}, {}\n".format(aFB, bR, cB))
        obj_file.write("v {}, {}, {}\n".format(aFT, bR, cT))
        obj_file.write("v {}, {}, {}\n".format(aFT, bL, cT))
        # in y
        obj_file.write("v {}, {}, {}\n".format(aRB, bL, cB))
        obj_file.write("v {}, {}, {}\n".format(aFB, bL, cB))
        obj_file.write("v {}, {}, {}\n".format(aFT, bL, cT))
        obj_file.write("v {}, {}, {}\n".format(aRT, bL, cT))
        #
        obj_file.write("v {}, {}, {}\n".format(aRB, bR, cB))
        obj_file.write("v {}, {}, {}\n".format(aFB, bR, cB))
        obj_file.write("v {}, {}, {}\n".format(aFT, bR, cT))
        obj_file.write("v {}, {}, {}\n".format(aRT, bR, cT))   
        # in z
        obj_file.write("v {}, {}, {}\n".format(aRT, bL, cT))
        obj_file.write("v {}, {}, {}\n".format(aFT, bL, cT))
        obj_file.write("v {}, {}, {}\n".format(aFT, bR, cT))
        obj_file.write("v {}, {}, {}\n".format(aRT, bR, cT))
        #
        obj_file.write("v {}, {}, {}\n".format(aRB, bL, cB))
        obj_file.write("v {}, {}, {}\n".format(aFB, bL, cB))
        obj_file.write("v {}, {}, {}\n".format(aFB, bR, cB))
        obj_file.write("v {}, {}, {}\n".format(aRB, bR, cB))  
        # Face
        for i in range(6):
            obj_file.write("f {}, {}, {}, {}\n".format((dim+1)*i+1, (dim+1)*i+2, (dim+1)*i+3, (dim+1)*i+4))

        # Grains:
        OutputFile = open(dataPath + '/' + str(framecount) + '.csv', 'w+')
        OutputFile.write(
            'frame, time-step, part (0=excav), Depth[m], Px[m], Py[m], Pz[m], Vx[m/s], Vy[m/s], Vz[m/s], Lift[N], DrawbarPull[N], Runtime, PythonStepTime, VortexStepTime \n')
        OutputFile.write("{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}\n".format(application.getFrame(
        ), t, 0, excavDef.depth, excavPos[0], excavPos[1], excavPos[2], excavVel[0], excavVel[1], excavVel[2], lift, drawbarPull, (toc - tic), python_step_time, vortex_step_time))
        
        # Houdini: Partio
        bgeo_file = dataPath+'/'+str(framecount)+'.bgeo'
        particleSet = partio.create()
        P = particleSet.addAttribute("position", partio.VECTOR, 3)
        V = particleSet.addAttribute("velocity", partio.VECTOR, 3)
        id = particleSet.addAttribute("id", partio.INT, 1)
        particleSet.addParticles(numParticles)

        for grain in range(numParticles):
            grain_pos = grainPos[grain]
            if grain < numParticles_min:
                grain_vel = grainVel[grain]
            else:
                grain_vel = [0,0,0]
            OutputFile.write("{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}\n".format(application.getFrame(
            ), t, grain+1, grainDia[grain], grain_pos[0], grain_pos[1], grain_pos[2], grain_vel[0], grain_vel[1], grain_vel[2]))

            # Houdini: Partio
            particleSet.set(P, grain, (grain_pos[0], grain_pos[1], grain_pos[2]))
            particleSet.set(V, grain, (grain_vel[0], grain_vel[1], grain_vel[2]))
            particleSet.set(id, grain, (grain,))

        # Houdini: Partio
        partio.write(bgeo_file, particleSet)

        OutputFile.close()
        framecount = framecount+1


    # ------------------------------------------------------
    # Update Application 
    # ------------------------------------------------------
    # Simulation end conditions
    if (runType is 'fullRun') and (moving_excav is True):
        t_moving = t_moving+dt

    if step > estop:
        state = 'finished'
        print 'e-stop'

    if state is 'finished':
        toc = time.time()
        print 'runtime:', (toc - tic), 's'
        end = 1

    # Save particle positions as 'previous position' for velocity calculation at next time step
    TIC = time.time()
    if (step % SimFrames == SimFrames-1.0) or (step % check_state_update == check_state_update-1.0):
        previousGrainPos = []
        num = 0
        if grainPos:
            num = len(grainPos)
        for i in range(num):
            previousGrainPos.append(list(grainPos[i]))

    # Update dynamics in Vortex
    TICC = time.time()
    application.update()
    TOCC = time.time()

    # Print in Command Prompt
    print 'State =', state
    if moving_excav is True:
        print 'Excavation sim time =', t_moving, 's'
    else:
        print 'Soil prep sim time =', t, 's'
    vortex_step_time = (TOCC-TICC)

    # End of Loop
    t = t+dt
    step = step+1