import bpy
import mathutils

def extract_complete_camera_data(camera_name=None):
    """
    Extract complete camera data:
    1. Direct extraction (no calculation)
    2. World-space quaternion+translation
    3. Camera intrinsics
    """

    # Get camera object
    if camera_name is None:
        camera_obj = bpy.context.scene.camera
    else:
        camera_obj = bpy.data.objects.get(camera_name)

    if not camera_obj or camera_obj.type != 'CAMERA':
        print("Camera not found")
        return None

    camera_data = camera_obj.data
    scene = bpy.context.scene

    # ===== 1. DIRECT EXTRACTION (NO CALCULATION) =====
    direct_data = {
        'name': camera_obj.name,
        'matrix_world': camera_obj.matrix_world.copy(),
        'location': camera_obj.location.copy(),
        'rotation_euler': camera_obj.rotation_euler.copy(),
        'rotation_quaternion': camera_obj.rotation_quaternion.copy(),
        'scale': camera_obj.scale.copy(),
    }

    # ===== 2. WORLD-SPACE QUATERNION + TRANSLATION =====
    matrix_world = camera_obj.matrix_world.copy()
    world_quaternion = matrix_world.to_quaternion()
    world_translation = matrix_world.to_translation()

    pose_data = {
        'qw': world_quaternion.w,
        'qx': world_quaternion.x,
        'qy': world_quaternion.y,
        'qz': world_quaternion.z,
        'tx': world_translation.x,
        'ty': world_translation.y,
        'tz': world_translation.z,
    }

    # ===== 3. CAMERA INTRINSICS =====
    # Image dimensions
    res_x = scene.render.resolution_x * scene.render.resolution_percentage / 100
    res_y = scene.render.resolution_y * scene.render.resolution_percentage / 100

    # Calculate focal length in pixels
    if camera_data.sensor_fit == 'VERTICAL':
        sensor_size = camera_data.sensor_height
        image_size = res_y
    else:  # 'HORIZONTAL' or 'AUTO'
        sensor_size = camera_data.sensor_width
        image_size = res_x

    focal_length_px = camera_data.lens / sensor_size * image_size

    # Principal point (assuming centered)
    cx = res_x / 2.0
    cy = res_y / 2.0

    intrinsic_data = {
        'focal_length_mm': camera_data.lens,
        'focal_length_px': focal_length_px,
        'sensor_width_mm': camera_data.sensor_width,
        'sensor_height_mm': camera_data.sensor_height,
        'sensor_fit': camera_data.sensor_fit,
        'resolution_x': res_x,
        'resolution_y': res_y,
        'cx': cx,
        'cy': cy,
        'intrinsic_matrix': [
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ],
        'clip_start': camera_data.clip_start,
        'clip_end': camera_data.clip_end,
    }

    # Add FOV if perspective camera
    if camera_data.type == 'PERSP':
        intrinsic_data.update({
            'fov_radians': camera_data.angle,
            'fov_degrees': camera_data.angle * 57.2958,
            'fov_x_radians': camera_data.angle_x,
            'fov_y_radians': camera_data.angle_y,
        })

    return direct_data, pose_data, intrinsic_data

def write_camera_data_to_txt(filepath, camera_name=None):
    """
    Write complete camera data to TXT file
    """

    result = extract_complete_camera_data(camera_name)
    if not result:
        print("Failed to extract camera data")
        return False

    direct_data, pose_data, intrinsic_data = result

    try:
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"BLENDER CAMERA EXTRACTION: {direct_data['name']}\n")
            f.write("=" * 60 + "\n\n")

            # ===== SECTION 1: DIRECT EXTRACTION =====
            f.write("1. DIRECT EXTRACTION (NO CALCULATION)\n")
            f.write("-" * 40 + "\n")

            f.write(f"Camera Name: {direct_data['name']}\n\n")

            f.write("4x4 World Matrix (matrix_world):\n")
            matrix = direct_data['matrix_world']
            for i in range(4):
                f.write(f"[{matrix[i][0]:10.6f}, {matrix[i][1]:10.6f}, {matrix[i][2]:10.6f}, {matrix[i][3]:10.6f}]\n")
            f.write("\n")

            f.write("Object Location (location):\n")
            loc = direct_data['location']
            f.write(f"[{loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f}]\n\n")

            f.write("Object Rotation Euler (rotation_euler):\n")
            euler = direct_data['rotation_euler']
            f.write(f"[{euler.x:.6f}, {euler.y:.6f}, {euler.z:.6f}] radians\n")
            f.write(f"[{euler.x*57.2958:.2f}, {euler.y*57.2958:.2f}, {euler.z*57.2958:.2f}] degrees\n\n")

            f.write("Object Rotation Quaternion (rotation_quaternion):\n")
            quat = direct_data['rotation_quaternion']
            f.write(f"[{quat.w:.6f}, {quat.x:.6f}, {quat.y:.6f}, {quat.z:.6f}] (w,x,y,z)\n\n")

            f.write("Object Scale (scale):\n")
            scale = direct_data['scale']
            f.write(f"[{scale.x:.6f}, {scale.y:.6f}, {scale.z:.6f}]\n\n")

            # ===== SECTION 2: WORLD POSE =====
            f.write("2. WORLD-SPACE POSE (qw,qx,qy,qz,tx,ty,tz)\n")
            f.write("-" * 40 + "\n")

            f.write("World-space Quaternion + Translation:\n")
            f.write(f"qw: {pose_data['qw']:.6f}\n")
            f.write(f"qx: {pose_data['qx']:.6f}\n")
            f.write(f"qy: {pose_data['qy']:.6f}\n")
            f.write(f"qz: {pose_data['qz']:.6f}\n")
            f.write(f"tx: {pose_data['tx']:.6f}\n")
            f.write(f"ty: {pose_data['ty']:.6f}\n")
            f.write(f"tz: {pose_data['tz']:.6f}\n\n")

            f.write("Compact Format:\n")
            f.write(f"Quaternion [w,x,y,z]: [{pose_data['qw']:.6f}, {pose_data['qx']:.6f}, {pose_data['qy']:.6f}, {pose_data['qz']:.6f}]\n")
            f.write(f"Translation [x,y,z]:  [{pose_data['tx']:.6f}, {pose_data['ty']:.6f}, {pose_data['tz']:.6f}]\n\n")

            f.write("One-liner (qw,qx,qy,qz,tx,ty,tz):\n")
            f.write(f"[{pose_data['qw']:.6f}, {pose_data['qx']:.6f}, {pose_data['qy']:.6f}, {pose_data['qz']:.6f}, {pose_data['tx']:.6f}, {pose_data['ty']:.6f}, {pose_data['tz']:.6f}]\n\n")

            # ===== SECTION 3: INTRINSICS =====
            f.write("3. CAMERA INTRINSICS\n")
            f.write("-" * 40 + "\n")

            f.write("Lens Parameters:\n")
            f.write(f"Focal Length: {intrinsic_data['focal_length_mm']:.2f} mm\n")
            f.write(f"Focal Length: {intrinsic_data['focal_length_px']:.2f} pixels\n")
            f.write(f"Sensor Size: {intrinsic_data['sensor_width_mm']:.2f} x {intrinsic_data['sensor_height_mm']:.2f} mm\n")
            f.write(f"Sensor Fit: {intrinsic_data['sensor_fit']}\n\n")

            f.write("Image Parameters:\n")
            f.write(f"Resolution: {intrinsic_data['resolution_x']:.0f} x {intrinsic_data['resolution_y']:.0f} pixels\n")
            f.write(f"Principal Point: ({intrinsic_data['cx']:.2f}, {intrinsic_data['cy']:.2f})\n")
            f.write(f"Clip Planes: {intrinsic_data['clip_start']:.3f} to {intrinsic_data['clip_end']:.3f}\n\n")

            f.write("Intrinsic Matrix (K):\n")
            K = intrinsic_data['intrinsic_matrix']
            for row in K:
                f.write(f"[{row[0]:10.2f}, {row[1]:10.2f}, {row[2]:10.2f}]\n")
            f.write("\n")

            # Add FOV if available
            if 'fov_degrees' in intrinsic_data:
                f.write("Field of View:\n")
                f.write(f"FOV: {intrinsic_data['fov_degrees']:.2f} degrees ({intrinsic_data['fov_radians']:.4f} radians)\n")
                f.write(f"FOV X: {intrinsic_data['fov_x_radians']:.4f} radians\n")
                f.write(f"FOV Y: {intrinsic_data['fov_y_radians']:.4f} radians\n\n")

            # ===== COORDINATE SYSTEM INFO =====
            f.write("4. COORDINATE SYSTEM REFERENCE\n")
            f.write("-" * 40 + "\n")
            f.write("Blender Coordinate System:\n")
            f.write("- X-axis: Right\n")
            f.write("- Y-axis: Forward\n")
            f.write("- Z-axis: Up\n")
            f.write("- Camera looks down NEGATIVE local Z-axis\n")
            f.write("- Right-handed coordinate system\n\n")

        print(f"Camera data exported to: {filepath}")
        return True

    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def print_camera_summary(camera_name=None):
    """
    Print a quick summary to console
    """
    result = extract_complete_camera_data(camera_name)
    if not result:
        return

    direct_data, pose_data, intrinsic_data = result

    print(f"\n=== Camera Summary: {direct_data['name']} ===")

    print("\nDirect Location:", [f"{x:.3f}" for x in direct_data['location']])
    print("World Pose (qw,qx,qy,qz,tx,ty,tz):")
    print(f"  [{pose_data['qw']:.3f}, {pose_data['qx']:.3f}, {pose_data['qy']:.3f}, {pose_data['qz']:.3f}, {pose_data['tx']:.3f}, {pose_data['ty']:.3f}, {pose_data['tz']:.3f}]")
    print(f"Focal Length: {intrinsic_data['focal_length_mm']}mm ({intrinsic_data['focal_length_px']:.1f}px)")
    print(f"Resolution: {intrinsic_data['resolution_x']:.0f}x{intrinsic_data['resolution_y']:.0f}")

def export_all_cameras_txt(filepath_base):
    """
    Export all cameras to separate TXT files
    """
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']

    exported_count = 0
    for camera in cameras:
        camera_filepath = f"{filepath_base}_{camera.name}.txt"
        if write_camera_data_to_txt(camera_filepath, camera.name):
            exported_count += 1

    print(f"Exported {exported_count} cameras to TXT files")
    return exported_count

# Example usage
if __name__ == "__main__":
    print("Starting camera extraction...")

    # Specify camera name and output path
    camera_name = "xiangji"
    output_dir = "C:/Users/26679/Desktop/blendr"

    # Check if the specific camera exists
    camera_obj = bpy.data.objects.get(camera_name)
    if camera_obj is None or camera_obj.type != 'CAMERA':
        print(f"ERROR: Camera '{camera_name}' not found in the scene!")
        print("Available cameras in scene:")
        cameras = [obj.name for obj in bpy.data.objects if obj.type == 'CAMERA']
        if cameras:
            for cam in cameras:
                print(f"  - {cam}")
        else:
            print("  No cameras found in scene!")
    else:
        print(f"Found camera: {camera_name}")

        # Try to create the output directory
        import os
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Directory {output_dir} created/verified")
        except Exception as e:
            print(f"Could not create directory {output_dir}: {e}")

        # Print summary to console
        print_camera_summary(camera_name)

        # Export camera data to TXT file
        output_file = os.path.join(output_dir, f"{camera_name}_data.txt")
        try:
            success = write_camera_data_to_txt(output_file, camera_name)
            if success:
                print(f"SUCCESS: Camera data exported to {output_file}")
            else:
                print(f"FAILED: Could not export to {output_file}")
        except Exception as e:
            print(f"Export error: {e}")

        # Also export all cameras to the same directory
        try:
            count = export_all_cameras_txt(os.path.join(output_dir, "camera"))
            print(f"Exported {count} camera(s) to {output_dir}/")
        except Exception as e:
            print(f"Export all cameras error: {e}")

    print("Camera extraction completed.")
