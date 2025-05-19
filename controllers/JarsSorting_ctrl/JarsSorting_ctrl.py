"""jar_sorter_ctrl_tflite.py controller
    ** Using UR5e from Universal Robot
    ** Using TensorFlow Lite for jar detection
    ** Using numpy for array handling
"""

import cv2
import numpy as np
import os
# Attempt to import TFLite runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        print("ERROR: TensorFlow Lite runtime not found. Please install tflite-runtime or tensorflow.")
        Interpreter = None


from controller import Supervisor, Mouse

robot = Supervisor()
timestep = 32 # Using your original timestep

# Jar classification
JAR_TYPE_GOOD = 0
JAR_TYPE_DEFECTED = 1
JAR_TYPE_UNKNOWN = -1
jar_type_names = ['GoodJar', 'DefectedJar'] # Corresponds to JAR_TYPE_GOOD and JAR_TYPE_DEFECTED

# Jar counters
good_jar_count = 0
defected_jar_count = 0

# Conveyor belt power
pwr_cb = False
click = False # For mouse click state

# Delay and state machine
counter = 0 # General purpose delay counter for actions
hmi_timer = 0 # Timer for HMI updates
robot_state = 0 # 0: WAITING, 1: PICKING, 2: MOVING_TO_DROP, 3: DROPPING, 4: RETURNING

# UR5e Robot arm target positions (joint angles)
# You'll need to define these based on your scene and desired drop locations
# Example: target_positions_good_jar = [-1.57, -1.8, -2.1, -2.3, -1.5]
#          target_positions_defected_jar = [1.57, -1.8, -2.1, -2.3, -1.5] # Different pan angle
target_positions_map = {
    JAR_TYPE_GOOD: [-1.570796, -1.87972, -2.139774, -2.363176, -1.50971], # Original target
    JAR_TYPE_DEFECTED: [0.5, -1.87972, -2.139774, -2.363176, -1.50971] # Example: Adjusted pan for defected
}
home_position = [0.0, 0.0, 0.0, 0.0, 0.0] # Home position for the arm joints

# Speed of UR5e Robot
ur_speed = 2.0

# Gripper motors
hand_motors = [robot.getDevice(f'finger_{i}_joint_1') for i in ['1', '2', 'middle']]

# UR5e arm motors
ur_motor_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint']
ur_motors = [robot.getDevice(name) for name in ur_motor_names]

for motor in ur_motors:
    motor.setVelocity(ur_speed)

# Gripper distance sensor
distance_sensor = robot.getDevice('distance sensor') # Ensure DEF name is 'distance sensor'
distance_sensor.enable(timestep)

# Wrist position sensor (to check if arm reached target)
# Using wrist_1_joint_sensor as an example, can monitor other joints or use a group
wrist_sensor_name = 'wrist_1_joint_sensor' # Ensure this sensor exists
position_sensor = robot.getDevice(wrist_sensor_name)
position_sensor.enable(timestep)

# Camera
camera = robot.getDevice('camera') # Ensure DEF name is 'camera'
camera.enable(timestep)
camera_width = camera.getWidth()
camera_height = camera.getHeight()

# Display for bounding boxes/classification
display = robot.getDevice('display') # Ensure DEF name is 'display'
display.attachCamera(camera)
display.setColor(0x00FF00) # Green for good, can change based on detection
display.setFont('Verdana', 16, True)

# HMI Display
hmi = robot.getDevice('hmi') # Ensure DEF name is 'hmi'
hmi.setColor(0x000000)
hmi.setFont('Verdana', 14, True)
hmi_image_id = 0 # For alternating HMI image files

# Speaker (optional)
# speaker = robot.getDevice('speaker')

# Mouse
robot.mouse.enable(timestep)

# Conveyor belt speed field
cb_node = robot.getFromDef('cb') # Ensure 'cb' is the DEF name of your conveyor belt
if cb_node:
    cb_speed_field = cb_node.getField('speed')
else:
    print("ERROR: Conveyor belt node 'cb' not found.")
    cb_speed_field = None

# Control panel URL field (if still used for HMI image)
cp_node = robot.getFromDef('CP') # Ensure 'CP' is the DEF name of your control panel display
if cp_node:
    cp_url_field = cp_node.getField('url')
else:
    print("WARNING: Control Panel node 'CP' not found for HMI.")
    cp_url_field = None

# --- TFLite Model Initialization ---
TFLITE_MODEL_PATH = "C:\Users\ABDOU\Desktop\ProjectSem2\Webots\Universal Robot V4 Python\yolov7_model.tflite" # CHANGE THIS TO YOUR MODEL PATH
TFLITE_LABELS_PATH = "C:\Users\ABDOU\Desktop\ProjectSem2\Webots\Universal Robot V4 Python\yolov7_label.tflite"  # CHANGE THIS TO YOUR LABELS FILE
tflite_interpreter = None
tflite_input_details = None
tflite_output_details = None
labels = []

def load_tflite_model():
    global tflite_interpreter, tflite_input_details, tflite_output_details, labels
    if not Interpreter:
        print("TFLite Interpreter not available. Detection will not work.")
        return False

    try:
        tflite_interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        tflite_interpreter.allocate_tensors()
        tflite_input_details = tflite_interpreter.get_input_details()
        tflite_output_details = tflite_interpreter.get_output_details()

        with open(TFLITE_LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        
        # Assuming labels are 'GoodJar', 'DefectedJar' in the labels file
        # and match jar_type_names
        if labels != jar_type_names:
            print(f"WARNING: Labels in {TFLITE_LABELS_PATH} ({labels}) do not match expected {jar_type_names}. Ensure order is correct.")

        print("TFLite model and labels loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading TFLite model or labels: {e}")
        return False

# --- Utility Functions ---
def convert_path(path):
    separator = os.path.sep
    return path.replace(separator, '/') if separator != '/' else path

def get_hmi_image_path(file_name_stem):
    # Modifies path to save HMI images in a 'textures' folder relative to 'worlds'
    path0 = os.path.abspath(os.getcwd())
    path0 = convert_path(path0)
    find0 = path0.find('controllers')
    if find0 != -1:
        path0 = path0[:find0]
        return path0 + 'worlds/textures/' + file_name_stem + '.jpg'
    else: # Fallback if structure is different
        return file_name_stem + '.jpg'

# def play_sound(jar_type_detected): # Optional
#     if speaker:
#         sound_file = ""
#         if jar_type_detected == JAR_TYPE_GOOD: sound_file = 'sounds/good_jar.wav' # Create these sounds
#         elif jar_type_detected == JAR_TYPE_DEFECTED: sound_file = 'sounds/defected_jar.wav'
        
#         if sound_file:
#             speaker.playSound(speaker, speaker, sound_file, 1.0, 1.0, 0.0, False)

def update_hmi_display():
    global hmi_image_id
    if not cp_url_field and not hmi: return # Nothing to update

    hmi_base_image_path = "hmi_jars_template.png" # Create a template image like your 'hmi.png'
    
    # Create image if template exists
    if os.path.exists(hmi_base_image_path):
        img = cv2.imread(hmi_base_image_path)
        if img is None:
            print(f"Error reading HMI template: {hmi_base_image_path}")
            return
    else: # Create a blank image if no template
        img = np.full((150, 200, 3), 255, dtype=np.uint8) # White background, adjust size
        cv2.putText(img, "HMI Display", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


    # Display counts
    cv2.putText(img, f'Good Jars: {good_jar_count:3d}', (30, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 0), 1)
    cv2.putText(img, f'Defected Jars: {defected_jar_count:3d}', (30, 120), cv2.FONT_HERSHEY_PLAIN, 1, (100, 0, 0), 1)

    # Conveyor status and time
    conveyor_status = 'ON ' if pwr_cb else 'OFF'
    cv2.putText(img, f'{conveyor_status} Time: {robot.getTime():.0f}s', (25, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # Save and display on HMI device
    hmi_temp_file_name = f'hmi_jars_rt{hmi_image_id}.png'
    cv2.imwrite(hmi_temp_file_name, img)
    
    if hmi:
        hmi.setAlpha(0.0)
        hmi.fillRectangle(0, 0, hmi.getWidth(), hmi.getHeight()) # Clear HMI device
        hmi.setAlpha(1.0)
        hmi_loaded_image = hmi.imageLoad(hmi_temp_file_name)
        hmi.imagePaste(hmi_loaded_image, 0, 0, False) # Paste new image
        hmi.imageDelete(hmi_loaded_image) # Clean up

    # Update 3D panel texture if CP node exists
    if cp_url_field:
        img_resized_for_panel = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        panel_texture_path_stem = f'hmi_panel_rt{hmi_image_id}'
        panel_texture_full_path_jpg = get_hmi_image_path(panel_texture_path_stem)
        cv2.imwrite(panel_texture_full_path_jpg, img_resized_for_panel)
        cp_url_field.setMFString(0, 'textures/' + panel_texture_path_stem + '.jpg')

    hmi_image_id = 1 - hmi_image_id # Alternate between 0 and 1
    if os.path.exists(hmi_temp_file_name): os.remove(hmi_temp_file_name) # Clean up temp png

def reset_on_screen_display():
    if display:
        display.setAlpha(0.0)
        display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
        display.setAlpha(1.0)

def draw_on_screen_display(label_text, x, y, w, h, color=0x00FF00): # x,y,w,h are relative to camera image
    if display:
        reset_on_screen_display()
        display.setColor(color)
        display.drawRectangle(x, y, w, h)
        display.drawText(label_text, x, y - 20 if y > 20 else y + h + 5)


def detect_jar_with_tflite():
    if not tflite_interpreter or not camera:
        return JAR_TYPE_UNKNOWN, None # Return unknown type and no bounding box

    # 1. Get image from camera
    image_data = camera.getImage()
    if not image_data:
        return JAR_TYPE_UNKNOWN, None

    # Convert to numpy array (BGRA) and then to RGB for model
    image_np = np.frombuffer(image_data, np.uint8).reshape((camera_height, camera_width, 4))
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)

    # 2. Preprocess image for TFLite model
    #    This depends HEAVILY on your TFLite model's requirements.
    #    Common steps: resize, normalize pixel values (e.g., to [0,1] or [-1,1]), expand_dims.
    input_shape = tflite_input_details[0]['shape'] # e.g., [1, 224, 224, 3]
    model_height, model_width = input_shape[1], input_shape[2]

    # ROI: Define a region of interest if jars always appear in a specific camera area
    # Example: roi = image_rgb[50:250, 50:250] # Adjust these values
    # For now, using the whole image resized
    img_resized = cv2.resize(image_rgb, (model_width, model_height))
    
    # Normalize (example for uint8 input, scale 0-1 if model expects that)
    # If your model expects float32 input in [0,1] range:
    # input_data = np.array(img_resized / 255.0, dtype=np.float32)
    # If your model expects uint8 input [0,255] (common with quantizied models):
    input_data = np.array(img_resized, dtype=np.uint8)

    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

    # 3. Run inference
    tflite_interpreter.set_tensor(tflite_input_details[0]['index'], input_data)
    tflite_interpreter.invoke()

    # 4. Postprocess output
    # This also depends on your model. Assuming a classification model for now.
    # Output is often an array of probabilities for each class.
    output_data = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])
    scores = output_data[0] # Assuming batch size 1

    detected_class_idx = np.argmax(scores)
    confidence = scores[detected_class_idx]

    # Define a confidence threshold
    CONFIDENCE_THRESHOLD = 0.6 # Adjust as needed

    # For debugging:
    # print(f"TFLite raw scores: {scores}, Highest: {labels[detected_class_idx]} with conf {confidence:.2f}")

    if confidence > CONFIDENCE_THRESHOLD:
        detected_label = labels[detected_class_idx]
        
        # Map label to JAR_TYPE
        if detected_label == jar_type_names[JAR_TYPE_GOOD]:
            current_jar_type = JAR_TYPE_GOOD
            display_color = 0x00FF00 # Green
        elif detected_label == jar_type_names[JAR_TYPE_DEFECTED]:
            current_jar_type = JAR_TYPE_DEFECTED
            display_color = 0xFF0000 # Red
        else:
            current_jar_type = JAR_TYPE_UNKNOWN
            display_color = 0xFFFF00 # Yellow for unknown recognized label
            print(f"Warning: Detected label '{detected_label}' not in known jar types.")
            return JAR_TYPE_UNKNOWN, None


        # Placeholder for bounding box - if your model provides it
        # For now, draw a generic box or skip.
        # If you have bbox from model: (xmin, ymin, xmax, ymax) normalized
        # box_x, box_y, box_w, box_h = 50, 50, camera_width-100, camera_height-100 # Example full box
        # For now, let's assume the object is somewhat centered if detected
        # We will draw a fixed-size rectangle for indication.
        # You should adapt this if your TFLite model provides bounding box coordinates.
        obj_w, obj_h = 80, 80 # Arbitrary size for display
        obj_x = (camera_width - obj_w) // 2
        obj_y = (camera_height - obj_h) // 2
        
        draw_on_screen_display(f"{detected_label} ({confidence:.2f})", obj_x, obj_y, obj_w, obj_h, display_color)
        return current_jar_type, [obj_x, obj_y, obj_w, obj_h] # Return type and a dummy box
    else:
        reset_on_screen_display() # Clear display if nothing confident
        return JAR_TYPE_UNKNOWN, None


# --- Robot Control Functions ---
def set_gripper(open_gripper):
    position = hand_motors[0].getMinPosition() if open_gripper else 0.52 # Adjust closed position (0.52 from your code)
    for motor in hand_motors:
        motor.setPosition(position)

def move_arm(target_joint_positions):
    for i, pos in enumerate(target_joint_positions):
        ur_motors[i].setPosition(pos)

def check_arm_at_position(target_joint_positions, tolerance=0.05):
    # Check one key joint, e.g., wrist or shoulder pan, for simplicity
    # A more robust check would verify all joints.
    # Using wrist_1_joint (index 3 for ur_motors, corresponding to wrist_1_joint_sensor)
    # Make sure position_sensor is for the joint you want to check predominantly
    # For instance, if target_positions_map changes shoulder_pan_joint significantly,
    # you might want a sensor on that, or check multiple sensors.
    # Current: checking wrist_1_joint_sensor against target_positions[3]
    current_pos = position_sensor.getValue()
    target_pos = target_joint_positions[ur_motor_names.index(wrist_sensor_name)] # Get target for the sensed joint
    return abs(current_pos - target_pos) < tolerance


# --- Main Loop ---
if not load_tflite_model():
    print("Failed to load TFLite model. Robot will not perform sorting.")
    # Optionally, stop the simulation or disable sorting logic
    # exit()

update_hmi_display() # Initial HMI update
set_gripper(True) # Start with gripper open
move_arm(home_position) # Go to home position initially
detected_jar_for_pickup = JAR_TYPE_UNKNOWN

while robot.step(timestep) != -1:
    if not cb_speed_field:
        print("Conveyor belt speed field not available. Exiting sorter.")
        break

    # Mouse control for conveyor
    mouse_state = robot.mouse.getState()
    # Adjust u,v values if your HMI button is at a different location
    if 0.026 < mouse_state.v < 0.059 and 0.017 < mouse_state.u < 0.047:
        if mouse_state.left and not click:
            pwr_cb = not pwr_cb
            click = True
            cb_speed_field.setSFFloat(0.15 if pwr_cb else 0.0) # Set speed from your original
    if not mouse_state.left and click:
        click = False

    if counter <= 0:
        # STATE 0: WAITING for a jar
        if robot_state == 0:
            move_arm(home_position) # Ensure arm is at home
            set_gripper(True) # Open gripper
            
            # Only try to detect if conveyor is on and arm is roughly home
            if pwr_cb and check_arm_at_position(home_position):
                jar_type, bbox = detect_jar_with_tflite()
                if jar_type != JAR_TYPE_UNKNOWN and distance_sensor.getValue() < 500: # Jar detected and close enough
                    detected_jar_for_pickup = jar_type
                    # play_sound(detected_jar_for_pickup) # Optional
                    
                    robot_state = 1 # PICKING
                    set_gripper(False) # Close gripper
                    counter = 15 # Wait for gripper to close (adjust as needed ~0.5s)
                    print(f"State: WAITING -> PICKING ({jar_type_names[detected_jar_for_pickup]})")


        # STATE 1: PICKING (Gripper closed, now lift and move)
        elif robot_state == 1:
            # This state implies gripper has closed. Now move to the drop location.
            # We assume the pick happens implicitly by the jar moving into the gripper at home.
            # If you need a dedicated picking motion (e.g., arm moving down), add it here or as a pre-state.
            if detected_jar_for_pickup != JAR_TYPE_UNKNOWN:
                target_drop_position = target_positions_map.get(detected_jar_for_pickup, home_position)
                move_arm(target_drop_position)
                robot_state = 2 # MOVING_TO_DROP
                print(f"State: PICKING -> MOVING_TO_DROP ({jar_type_names[detected_jar_for_pickup]})")
            else: # Should not happen if logic is correct
                robot_state = 0 # Go back to waiting
                print("Error: Tried to pick unknown jar type.")


        # STATE 2: MOVING_TO_DROP (Arm is moving to drop-off point)
        elif robot_state == 2:
            target_drop_position = target_positions_map.get(detected_jar_for_pickup, home_position)
            if check_arm_at_position(target_drop_position):
                robot_state = 3 # DROPPING
                set_gripper(True) # Open gripper to drop
                counter = 15 # Wait for gripper to open and jar to fall (adjust ~0.5s)
                print(f"State: MOVING_TO_DROP -> DROPPING ({jar_type_names[detected_jar_for_pickup]})")
                
                # Update counts
                if detected_jar_for_pickup == JAR_TYPE_GOOD:
                    good_jar_count += 1
                elif detected_jar_for_pickup == JAR_TYPE_DEFECTED:
                    defected_jar_count += 1
                update_hmi_display() # Update HMI with new counts
                reset_on_screen_display() # Clear detection display

        # STATE 3: DROPPING (Gripper opened, wait, then return arm)
        elif robot_state == 3:
            move_arm(home_position) # Start returning to home
            robot_state = 4 # RETURNING
            print("State: DROPPING -> RETURNING")
            detected_jar_for_pickup = JAR_TYPE_UNKNOWN # Reset for next cycle

        # STATE 4: RETURNING (Arm is moving back to home position)
        elif robot_state == 4:
            if check_arm_at_position(home_position):
                robot_state = 0 # WAITING (ready for next jar)
                print("State: RETURNING -> WAITING (Ready)")
                set_gripper(True) # Ensure gripper is open for next detection
    else:
        counter -= 1

    # HMI update periodically
    hmi_timer += 1
    if hmi_timer >= 10: # Update HMI every 10 steps (adjust for desired refresh rate)
        update_hmi_display()
        hmi_timer = 0
    
    # Pass