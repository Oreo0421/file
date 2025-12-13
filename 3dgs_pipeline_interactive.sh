#!/bin/bash

# 3DGS Animation Pipeline Interactive Script
# This script guides you through the entire pipeline with interactive prompts

set -e  # Exit on error

# Conda environment names
ANIM_ENV="anim3dgs"  # For AnimatableGaussians (Steps 0, 1)
HUGS_ENV="hugs"      # For ML-HUGS (Steps 2, 3)

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Initialize conda
eval "$(conda shell.bash hook)"

# Function to switch conda environment
switch_conda_env() {
    local env_name=$1
    echo -e "${CYAN}Switching to conda environment: $env_name${NC}"
    conda activate "$env_name"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully activated $env_name${NC}"
        return 0
    else
        print_error "Failed to activate conda environment: $env_name"
        return 1
    fi
}

# Function to check conda environment exists
check_conda_env() {
    local env_name=$1
    if conda env list | grep -q "^$env_name "; then
        return 0
    else
        return 1
    fi
}

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

# Function to print step info
print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
}

# Function to print errors
print_error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
}

# Function to ask yes/no questions
ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Function to get directory input with validation
get_directory() {
    local prompt="$1"
    local default="$2"
    local dir
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " dir
        dir=${dir:-$default}
    else
        read -p "$prompt: " dir
    fi
    
    echo "$dir"
}

# Function to get file input with validation
get_file() {
    local prompt="$1"
    local default="$2"
    local file
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " file
        file=${file:-$default}
    else
        read -p "$prompt: " file
    fi
    
    echo "$file"
}

# Function to get number input
get_number() {
    local prompt="$1"
    local default="$2"
    local num
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " num
        num=${num:-$default}
    else
        read -p "$prompt: " num
    fi
    
    echo "$num"
}

# Main script starts here
clear
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      3DGS Animation Pipeline - Interactive Script             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"

print_warning "This script will guide you through the entire pipeline."
print_warning "Please have all necessary paths and parameters ready."
echo ""

# Check conda environments
print_header "Checking Conda Environments"

echo "Checking required conda environments..."
echo ""

if check_conda_env "$ANIM_ENV"; then
    echo -e "${GREEN}✓ Found conda environment: $ANIM_ENV${NC}"
else
    print_error "Conda environment not found: $ANIM_ENV"
    echo "Please create it first with: conda create -n $ANIM_ENV python=3.x"
    exit 1
fi

if check_conda_env "$HUGS_ENV"; then
    echo -e "${GREEN}✓ Found conda environment: $HUGS_ENV${NC}"
else
    print_error "Conda environment not found: $HUGS_ENV"
    echo "Please create it first with: conda create -n $HUGS_ENV python=3.x"
    exit 1
fi

echo ""
echo -e "${CYAN}Pipeline will use:${NC}"
echo -e "  • ${CYAN}$ANIM_ENV${NC} for AnimatableGaussians (Steps 0-1)"
echo -e "  • ${CYAN}$HUGS_ENV${NC} for ML-HUGS (Steps 2-3)"
echo ""

# ============================================================================
# STEP 0: Initial Setup and Configuration
# ============================================================================

print_header "STEP 0: Initial Setup"

# Switch to AnimatableGaussians environment
switch_conda_env "$ANIM_ENV"
echo ""

echo "Do you need to generate a new animation or use existing results?"
echo "1) Generate new animation (run main_avatar_joint.py)"
echo "2) Use existing animation results"
read -p "Select option (1/2): " initial_choice

if [ "$initial_choice" == "1" ]; then
    print_step "0.1" "Generate New Animation"
    
    config_file=$(get_file "Enter path to avatar.yaml config file" "configs/avatarrex_zzr/avatar.yaml")
    
    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        exit 1
    fi
    
    echo ""
    print_warning "Before running, please update the following in $config_file:"
    echo "  - Line 69: pose_sequence npz file path"
    echo "  - Line 69: save_ply setting"
    echo ""
    
    if ask_yes_no "Have you updated the config file?"; then
        print_step "0.1" "Running main_avatar_joint.py..."
        python main_avatar_joint.py -c "$config_file" --mode=test
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Animation generation completed successfully${NC}"
        else
            print_error "Animation generation failed"
            exit 1
        fi
    else
        echo "Please update the config file and run this script again."
        exit 0
    fi
fi

# ============================================================================
# STEP 1: Process Joint Data
# ============================================================================

print_header "STEP 1: Process Joint Data"

# Ensure we're in AnimatableGaussians environment
switch_conda_env "$ANIM_ENV"
echo ""

print_step "1.1" "Save and Transform Joints"

input_joints_dir=$(get_directory "Enter input joints directory" \
    "/home/fzhi/fzt/3dgs_pipeline/animatable_3DGS/AnimatableGaussians/test_results/avatarrex_zzr/avatar/thuman4__pose_00_free_view/batch_700000/pca_20_sigma_2.00/joints/")

output_joints_dir=$(get_directory "Enter output joints directory" \
    "/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/output_combine/joint/")

if [ ! -d "$input_joints_dir" ]; then
    print_error "Input joints directory not found: $input_joints_dir"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_joints_dir"

print_step "1.1" "Running save_joints_trans_sequence.py..."
python save_joints_trans_sequence.py \
    --input_dir "$input_joints_dir" \
    --output_dir "$output_joints_dir"

if [ $? -ne 0 ]; then
    print_error "Failed to save joints"
    exit 1
fi

echo -e "${GREEN}✓ Joints saved successfully${NC}"

# ============================================================================
print_step "1.2" "Add Lines to Joint Sequence"

joint_npy_dir=$(get_directory "Enter transformed NPY directory" \
    "$output_joints_dir/npy/transformed/")

joint_pt_dir=$(get_directory "Enter output PT directory" \
    "$output_joints_dir/pt/")

# Determine frame range
echo ""
echo "Determining frame range..."
if [ -d "$joint_npy_dir" ]; then
    # Find the maximum frame number
    max_frame=$(ls "$joint_npy_dir"/*.npy 2>/dev/null | sed 's/.*_\([0-9]*\)\.npy/\1/' | sort -n | tail -1)
    if [ -n "$max_frame" ]; then
        # Remove leading zeros for display
        max_frame_display=$((10#$max_frame))
        echo -e "${GREEN}✓ Detected maximum frame: $max_frame_display${NC}"
        default_end=$max_frame_display
    else
        default_end=99
    fi
else
    default_end=99
fi

start_frame=$(get_number "Enter start frame" "0")
end_frame=$(get_number "Enter end frame" "$default_end")

mkdir -p "$joint_pt_dir"

print_step "1.2" "Running add_line2joint_sequence.py..."
python add_line2joint_sequence.py \
    --in_dir "$joint_npy_dir" \
    --out_dir "$joint_pt_dir" \
    --start "$start_frame" \
    --end "$end_frame"

if [ $? -ne 0 ]; then
    print_error "Failed to add lines to joints"
    exit 1
fi

echo -e "${GREEN}✓ Joint lines added successfully${NC}"

# ============================================================================
# STEP 2: Transform Human Gaussians
# ============================================================================

print_header "STEP 2: Transform Human Gaussians"

# Switch to ML-HUGS environment
switch_conda_env "$HUGS_ENV"
echo ""

print_step "2.1" "Configure Transformation Matrix"

trans_matrix_file=$(get_file "Enter path to transformation matrix file" \
    "/home/fzhi/fzt/3dgs_pipeline/animatable_dataset/human_trans/trans.md")

if [ -f "$trans_matrix_file" ]; then
    echo ""
    echo "Current transformation matrix:"
    cat "$trans_matrix_file"
    echo ""
fi

transform_script=$(get_file "Enter path to transform_human_sequence.py" \
    "/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/scripts/transform_human_sequence.py")

if [ ! -f "$transform_script" ]; then
    print_error "Transform script not found: $transform_script"
    exit 1
fi

print_warning "Please update the transform_matrix in $transform_script"
if ! ask_yes_no "Have you updated the transformation matrix?"; then
    echo "Please update the transformation matrix and run this script again."
    exit 0
fi

print_step "2.2" "Transform Human Sequence"

input_gaussians_dir=$(get_directory "Enter input posed_gaussians directory" \
    "/home/fzhi/fzt/3dgs_pipeline/animatable_3DGS/AnimatableGaussians/test_results/avatarrex_zzr/avatar/thuman4__pose_02_free_view/batch_700000/pca_20_sigma_2.00/posed_gaussians")

output_human_dir=$(get_directory "Enter output human directory" \
    "/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/output_human/thhumanpose02")

output_format=$(get_file "Enter output format (both/ply/pt)" "both")

human_start_frame=$(get_number "Enter start frame (e.g., 00002000)" "00002000")
human_end_frame=$(get_number "Enter end frame (e.g., 00002499)" "00002499")

mkdir -p "$output_human_dir"

print_step "2.2" "Running transform_human_sequence.py..."
python "$transform_script" \
    --input_dir "$input_gaussians_dir" \
    --output_dir "$output_human_dir" \
    --output_format "$output_format" \
    --start_frame "$human_start_frame" \
    --end_frame "$human_end_frame"

if [ $? -ne 0 ]; then
    print_error "Failed to transform human sequence"
    exit 1
fi

echo -e "${GREEN}✓ Human sequence transformed successfully${NC}"

# ============================================================================
# STEP 3: Combine Human and Scene Gaussians
# ============================================================================

print_header "STEP 3: Combine Human and Scene Gaussians"

# Ensure we're in ML-HUGS environment
switch_conda_env "$HUGS_ENV"
echo ""

print_step "3.1" "Render Combined Sequence"

human_pt_dir_render=$(get_directory "Enter human PT directory" \
    "$output_human_dir/pt")

scene_pt_file=$(get_file "Enter scene PT file path" \
    "/home/fzhi/fzt/3dgs_pipeline/animatable_dataset/scene/djr/djr_3dgs.pt")

output_combine_dir=$(get_directory "Enter output combine directory" \
    "/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/output_combine/")

camera_json=$(get_file "Enter camera JSON file path" \
    "/home/fzhi/fzt/3dgs_pipeline/ml_hug/ml-hugs/camera/djr_1m.json")

render_mode=$(get_file "Enter render mode (human_scene/human/scene)" "human_scene")

# Use the same frame range as transformation
render_start_frame=$(get_number "Enter start frame for rendering" "$human_start_frame")
render_end_frame=$(get_number "Enter end frame for rendering" "$human_end_frame")

if [ ! -f "$scene_pt_file" ]; then
    print_error "Scene PT file not found: $scene_pt_file"
    exit 1
fi

if [ ! -f "$camera_json" ]; then
    print_error "Camera JSON file not found: $camera_json"
    exit 1
fi

mkdir -p "$output_combine_dir"

print_step "3.1" "Running render_sequence_firstcamera.py..."
python hugs/renderer/render_sequence_firstcamera.py \
    --human_pt_dir "$human_pt_dir_render" \
    --scene_pt "$scene_pt_file" \
    --output_dir "$output_combine_dir" \
    --start_frame "$render_start_frame" \
    --end_frame "$render_end_frame" \
    --camera_json "$camera_json" \
    --render_mode "$render_mode"

if [ $? -ne 0 ]; then
    print_error "Failed to render combined sequence"
    exit 1
fi

echo -e "${GREEN}✓ Combined sequence rendered successfully${NC}"

# ============================================================================
# Final Summary
# ============================================================================

print_header "Pipeline Completed Successfully!"

echo -e "${GREEN}✓ All steps completed successfully!${NC}"
echo ""
echo "Output locations:"
echo "  - Joints (PT): $joint_pt_dir"
echo "  - Human (transformed): $output_human_dir"
echo "  - Combined renders: $output_combine_dir"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review the output files in the directories above"
echo "  2. Check the rendered images/videos in: $output_combine_dir"
echo "  3. If needed, adjust parameters and re-run specific steps"
echo ""

# Ask if user wants to save the configuration
if ask_yes_no "Would you like to save this configuration for future use?"; then
    config_save_path=$(get_file "Enter path to save configuration" "./pipeline_config.txt")
    
    cat > "$config_save_path" << EOF
# 3DGS Pipeline Configuration
# Generated: $(date)

[Step 1 - Joint Processing]
input_joints_dir=$input_joints_dir
output_joints_dir=$output_joints_dir
joint_npy_dir=$joint_npy_dir
joint_pt_dir=$joint_pt_dir
start_frame=$start_frame
end_frame=$end_frame

[Step 2 - Human Transformation]
input_gaussians_dir=$input_gaussians_dir
output_human_dir=$output_human_dir
output_format=$output_format
human_start_frame=$human_start_frame
human_end_frame=$human_end_frame
transform_script=$transform_script

[Step 3 - Rendering]
human_pt_dir_render=$human_pt_dir_render
scene_pt_file=$scene_pt_file
output_combine_dir=$output_combine_dir
camera_json=$camera_json
render_mode=$render_mode
render_start_frame=$render_start_frame
render_end_frame=$render_end_frame
EOF
    
    echo -e "${GREEN}✓ Configuration saved to: $config_save_path${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Pipeline Complete!                         ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
