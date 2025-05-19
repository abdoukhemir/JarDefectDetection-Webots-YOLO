"""jar_spawner_ctrl.py controller."""

from controller import Supervisor
import random

random.seed()

robot = Supervisor()

timestep = 64

# Counters for jars
good_jar_count_spawned = 0
defected_jar_count_spawned = 0
current_good_jar_idx = 1 # To create unique DEF names like GoodJar1, GoodJar2
current_defected_jar_idx = 1 # To create unique DEF names like DefectedJar1, DefectedJar2

# Maximum number of each jar to spawn
max_good_jars = 25 # Adjust as needed
max_defected_jars = 25 # Adjust as needed

# Spawning logic
spawn_interval_steps = 120 # Corresponds to i == 120 in your original code
current_step_timer = 0
initial_spawn_delay_steps = 8 * (1000 // timestep) # Wait for ~8 seconds before starting to spawn, similar to 'timer > 8'

# Timer for initial delay
gate_timer = 0
time_benchmark = robot.getTime()

# Initialize gate motor
gate = robot.getDevice('gate')

# Initialize IR sensor of the gate
ir0 = robot.getDevice('ir0')
ir0.enable(timestep)

# Initialize LED of the gate
led = robot.getDevice('led')
led.set(1) # Assuming LED ON means gate is closed or ready

# Initial position for new jars
jar_initial_translation = [0.570002, 2.85005, 0.349962] # Use your original fruit_initial_translation

# Getting the field of the conveyor belt speed
cb_node = robot.getFromDef('cb') # Ensure 'cb' is the DEF name of your conveyor belt
if cb_node:
    cb_speed_field = cb_node.getField('speed')
else:
    print("ERROR: Conveyor belt node 'cb' not found.")
    cb_speed_field = None

# Getting the field of the control panel url (if still used)
# cp_node = robot.getFromDef('CP')
# cp_url_field = cp_node.getField('url')

def control_gate(current_speed):
    """Controls the gate based on IR sensor and conveyor speed."""
    if ir0.getValue() > 360 or current_speed == 0: # Adjust IR threshold if needed
        gate.setPosition(0) # Closed position
        led.set(1) # LED ON
    else:
        gate.setPosition(-0.349066) # Open position (use your original value)
        led.set(0) # LED OFF

# Main loop:
while robot.step(timestep) != -1:
    if not cb_speed_field:
        print("Conveyor belt speed field not available. Exiting spawner.")
        break

    # Get the conveyor belt speed
    speed = cb_speed_field.getSFFloat()

    control_gate(speed)

    if speed > 0:
        # Initial delay logic
        if int(robot.getTime()) != time_benchmark:
            time_benchmark = int(robot.getTime())
            gate_timer += 1

        if gate_timer > (initial_spawn_delay_steps // (1000 // timestep)): # Check if initial delay has passed
            current_step_timer += 1
            if current_step_timer >= spawn_interval_steps:
                current_step_timer = 0 # Reset spawn interval timer

                can_spawn_good = good_jar_count_spawned < max_good_jars
                can_spawn_defected = defected_jar_count_spawned < max_defected_jars

                if can_spawn_good or can_spawn_defected:
                    # Choose which type of jar to spawn
                    if can_spawn_good and can_spawn_defected:
                        choice = random.choice(['GoodJar', 'DefectedJar'])
                    elif can_spawn_good:
                        choice = 'GoodJar'
                    else:
                        choice = 'DefectedJar'

                    jar_node = None
                    if choice == 'GoodJar':
                        # IMPORTANT: Ensure you have PROTO files named 'GoodJar.proto' and 'DefectedJar.proto'
                        # or that 'GoodJar' and 'DefectedJar' are valid node types Webots can spawn.
                        # The DEF name needs to be unique for each instance spawned.
                        jar_def_name = f"SpawnedGoodJar{current_good_jar_idx}"
                        jar_string = f'DEF {jar_def_name} GoodJar {{ translation {jar_initial_translation[0]} {jar_initial_translation[1]} {jar_initial_translation[2]} }}'
                        good_jar_count_spawned += 1
                        current_good_jar_idx += 1
                    else: # DefectedJar
                        jar_def_name = f"SpawnedDefectedJar{current_defected_jar_idx}"
                        jar_string = f'DEF {jar_def_name} DefectedJar {{ translation {jar_initial_translation[0]} {jar_initial_translation[1]} {jar_initial_translation[2]} }}'
                        defected_jar_count_spawned += 1
                        current_defected_jar_idx += 1

                    # Get the root node and import the jar string as its last child
                    root_node = robot.getRoot()
                    root_children_field = root_node.getField('children')
                    root_children_field.importMFNodeFromString(-1, jar_string)
                    print(f"Spawned: {jar_def_name}")

                elif not can_spawn_good and not can_spawn_defected:
                    print("Maximum number of all jars spawned.")
                    # Optionally, stop the simulation or conveyor here
                    # cb_speed_field.setSFFloat(0.0) # Stop conveyor
                    pass
    # Pass