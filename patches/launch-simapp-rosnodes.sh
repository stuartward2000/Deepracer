#!/bin/bash
set -e
echo 'Use this script to launch a SageMaker training job.'

RANDOM_STR="$RANDOM-$(date +%s)"

echo '----------'
echo "EVALUATION: $EVALUATION"
echo "EVAL_WORLD_NAME: $EVAL_WORLD_NAME"
echo '----------'

if [ -z "$EVALUATION" ]; then
    WORLD_NAME=$(
        cat /configs/environment_params.yaml \
        | yq .WORLD_NAME
    )
    echo "Running training mode with ${WORLD_NAME} track."
elif [ "$EVALUATION" = 'true' ]; then
    WORLD_NAME="$EVAL_WORLD_NAME"
    echo "Running evaluation mode with ${WORLD_NAME} track."
else
    WORLD_NAME=$(
        cat /configs/environment_params.yaml \
        | yq .WORLD_NAME
    )
    echo "Running training mode with ${WORLD_NAME} track."
fi

NUMBER_OF_OBSTACLES=$(
    cat /configs/environment_params.yaml \
    | yq .NUMBER_OF_OBSTACLES
)
NUMBER_OF_BOT_CARS=$(
    cat /configs/environment_params.yaml \
    | yq .NUMBER_OF_BOT_CARS
)

AWS_REGION=us-east-1
BASE_JOBNAME="rlexp-deepracer"
SM_JOBNAME="${BASE_JOBNAME}-prefix"
S3_PREFIX="sagemaker-${SM_JOBNAME}"

PRETRAINED_MODEL="" # for cloning

S3_BUCKET="aws-deepracer-bba2e912-6ef0-4c3c-a072-ce17e254bcf2"
while [[ -z "$S3_BUCKET" ]]; do
    echo -n "Enter DeepRacer S3 Bucket Name: "
    read S3_BUCKET
done

S3_ROS_LOG_BUCKET="aws-deepracer-bba2e912-6ef0-4c3c-a072-ce17e254bcf2"
while [[ -z "$S3_ROS_LOG_BUCKET" ]]; do
    echo -n "Default S3_ROS_LOG_BUCKET S3 Bucket Name to ${S3_BUCKET} "
    S3_ROS_LOG_BUCKET=S3_BUCKET
done

echo "Job name: ${SM_JOBNAME}."

# mkdir does not work with nested directories in apptainer for some reason
mkdir /${S3_BUCKET}
mkdir /${S3_BUCKET}/${S3_PREFIX}
mkdir /${S3_BUCKET}/${S3_PREFIX}/model

# Upload the reward function & model metadata
REWARD_FUNCTION_S3_KEY=${S3_PREFIX}/custom_reward_function.py
MODEL_METADATA_S3_KEY=${S3_PREFIX}/model/model_metadata.json

REWARD_FUNCTION_S3_SOURCE=/${S3_BUCKET}/${REWARD_FUNCTION_S3_KEY}
MODEL_METADATA_S3_SOURCE=/${S3_BUCKET}/${MODEL_METADATA_S3_KEY}

cp /configs/agent_params.json ${MODEL_METADATA_S3_SOURCE}
echo "Uploaded model_metadata to ${MODEL_METADATA_S3_SOURCE}"
cp /configs/reward_function.py ${REWARD_FUNCTION_S3_SOURCE}
echo "Uploaded reward function to ${REWARD_FUNCTION_S3_SOURCE}"

# generate local yaml file and then upload to S3 bucket for robomaker training
DEFAULT_YAML="default_training_params.yaml"
touch ${DEFAULT_YAML}
echo "WORLD_NAME:                           \"${WORLD_NAME}\"" | tee ${DEFAULT_YAML}
echo "SAGEMAKER_SHARED_S3_BUCKET:           \"${S3_BUCKET}\"" | tee -a ${DEFAULT_YAML}
echo "SAGEMAKER_SHARED_S3_PREFIX:           \"${S3_PREFIX}\"" | tee -a ${DEFAULT_YAML}
echo "TRAINING_JOB_ARN:                     \"${SAGEMAKER_TRAINING_JOB_ARN}\"" | tee -a ${DEFAULT_YAML}
echo "METRICS_S3_BUCKET:                    \"${S3_BUCKET}\"" | tee -a ${DEFAULT_YAML}
echo "METRICS_S3_OBJECT_KEY:                \"/logs/training_metrics.json\"" | tee -a ${DEFAULT_YAML}
echo "SIMTRACE_S3_BUCKET:                   \"${S3_BUCKET}\"" | tee -a ${DEFAULT_YAML}
echo "SIMTRACE_S3_PREFIX:                   \"${S3_PREFIX}/iteration-data/training\"" | tee -a ${DEFAULT_YAML}
echo "MP4_S3_BUCKET:                        \"${S3_BUCKET}\"" | tee -a ${DEFAULT_YAML}
echo "MP4_S3_OBJECT_PREFIX:                 \"${S3_PREFIX}/iteration-data/training\"" | tee -a ${DEFAULT_YAML}
echo "AWS_REGION:                           \"${AWS_REGION}\"" | tee -a ${DEFAULT_YAML}
echo "TARGET_REWARD_SCORE:                  \"None\"" | tee -a ${DEFAULT_YAML}
echo "NUMBER_OF_EPISODES:                   \"0\"" | tee -a ${DEFAULT_YAML}
echo "JOB_TYPE:                             \"TRAINING\"" | tee -a ${DEFAULT_YAML}
echo "CHANGE_START_POSITION:                \"true\"" | tee -a ${DEFAULT_YAML}
echo "ALTERNATE_DRIVING_DIRECTION:          \"true\"" | tee -a ${DEFAULT_YAML}
echo "REWARD_FILE_S3_KEY:                   \"${REWARD_FUNCTION_S3_KEY}\"" | tee -a ${DEFAULT_YAML}
echo "MODEL_METADATA_FILE_S3_KEY:           \"${MODEL_METADATA_S3_KEY}\"" | tee -a ${DEFAULT_YAML}
echo "NUMBER_OF_OBSTACLES:                  \"${NUMBER_OF_OBSTACLES}\"" | tee -a ${DEFAULT_YAML}
echo "IS_OBSTACLE_BOT_CAR:                  \"false\"" | tee -a ${DEFAULT_YAML}
echo "RANDOMIZE_OBSTACLE_LOCATIONS:         \"true\"" | tee -a ${DEFAULT_YAML}
# echo "OBJECT_POSITIONS:
#  - 0.1690708037909166, -1
#  - 0.2638102569075569, 1
#  - 0.4072827740044651, -1
#  - 0.5804718430735435, 1
#  - 0.6937442410812812, -1
#  - 0.7864867324330095, 1" | tee -a ${DEFAULT_YAML}
echo "IS_LANE_CHANGE:                       \"false\"" | tee -a ${DEFAULT_YAML}
echo "LOWER_LANE_CHANGE_TIME:               \"3.0\"" | tee -a ${DEFAULT_YAML}
echo "UPPER_LANE_CHANGE_TIME:               \"5.0\"" | tee -a ${DEFAULT_YAML}
echo "LANE_CHANGE_DISTANCE:                 \"1.0\"" | tee -a ${DEFAULT_YAML}
echo "NUMBER_OF_BOT_CARS:                   \"${NUMBER_OF_BOT_CARS}\"" | tee -a ${DEFAULT_YAML}
echo "MIN_DISTANCE_BETWEEN_BOT_CARS:        \"2.0\"" | tee -a ${DEFAULT_YAML}
echo "RANDOMIZE_BOT_CAR_LOCATIONS:          \"true\"" | tee -a ${DEFAULT_YAML}
echo "BOT_CAR_SPEED:                        \"0.2\"" | tee -a ${DEFAULT_YAML}
echo "CAR_COLOR:                            \"Blue\"" | tee -a ${DEFAULT_YAML}
echo "NUMBER_OF_RESETS:                     \"0\"" | tee -a ${DEFAULT_YAML}
echo "RACE_TYPE:                            \"HEAD_TO_BOT\"" | tee -a ${DEFAULT_YAML}
echo "ENABLE_DOMAIN_RANDOMIZATION:          \"false\"" | tee -a ${DEFAULT_YAML}
echo "DISPLAY_NAME:                         \"LongLongRacerNameBlaBlaBla\"" | tee -a ${DEFAULT_YAML}
echo "REVERSE_DIR:                          \"false\"" | tee -a ${DEFAULT_YAML}
echo "BODY_SHELL_TYPE:                      \"deepracer\"" | tee -a ${DEFAULT_YAML}
echo "IS_CONTINUOUS:                        \"true\"" | tee -a ${DEFAULT_YAML}
echo "LEADERBOARD_NAME:                     \"cs7642\"" | tee -a ${DEFAULT_YAML}

NUM_WORKERS=1
echo "NUM_WORKERS:                          \"${NUM_WORKERS}\"" | tee -a ${DEFAULT_YAML}

SOURCE_YAML="/configs/environment_params.yaml"
S3_YAML_NAME="training_params.yaml"

if [ -z "$EVALUATION" ]; then
    yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' ${DEFAULT_YAML} ${SOURCE_YAML} | tee ${S3_YAML_NAME}
elif [ "$EVALUATION" = 'true' ]; then
    cp ${DEFAULT_YAML} ${S3_YAML_NAME}
else
    yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' ${DEFAULT_YAML} ${SOURCE_YAML} | tee ${S3_YAML_NAME}
fi

YAML_S3_KEY=${S3_PREFIX}/${S3_YAML_NAME}
YAML_S3_SOURCE=/${S3_BUCKET}/${YAML_S3_KEY}
cp ${S3_YAML_NAME} ${YAML_S3_SOURCE}
echo "Uploaded training params to ${YAML_S3_SOURCE}"


echo "Starting sageonly.sh"

COACH_EXP_NAME=sagemaker_rl

# ros melodic
export ROS_DISTRO=melodic
export PYTHONUNBUFFERED=1
export XAUTHORITY=/root/.Xauthority

IP_ADDRESSES=$(hostname -I)
echo "HOSTNAME -I ${IP_ADDRESSES}"

# Set space as the delimiter
IFS=' '
# Read the split words into an array based on space delimiter
read -a IPS_ADDRESS_LIST <<< "$IP_ADDRESSES"
unset IFS
export ROS_IP=${IPS_ADDRESS_LIST[0]}
echo "Using ROS IP ${ROS_IP}"

unset KINESIS_VIDEO_STREAM_NAME

export APP_REGION=${AWS_REGION}
export MODEL_S3_BUCKET=${S3_BUCKET}
export MODEL_S3_PREFIX=${S3_PREFIX}
export S3_YAML_NAME=${S3_YAML_NAME}
export WORLD_NAME=${WORLD_NAME}
export SIMULATION_LAUNCH_FILE='distributed_training.launch'

export SAGEMAKER_SHARED_S3_BUCKET=${S3_BUCKET}
export SAGEMAKER_SHARED_S3_PREFIX=${S3_PREFIX}

export DEEPRACER_JOB_TYPE_ENV="SAGEONLY"

export PYTHONPATH=/opt/amazon/install/sagemaker_rl_agent/lib/python3.6/site-packages/:$PYTHONPATH

export PATH="/opt/ml/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
if which x11vnc &>/dev/null; then
    source /opt/ros/$ROS_DISTRO/setup.bash
    source /opt/amazon/install/setup.bash
    export GAZEBO_MODEL_PATH='/opt/amazon/install/deepracer_simulation_environment/share/deepracer_simulation_environment'
    
    # select random display to avoid conflicts
    # check which one is free with: ps aux | grep X
    export DISPLAY=":$(( RANDOM % 99 + 1 ))"
    Xvfb "$DISPLAY" -ac -screen 0 1400x900x24 &
    echo "Using DISPLAY=${DISPLAY}"
    
    echo "Running simulation job on single sagemaker instance..."
    echo "Check ${SIMULATION_LOG_GROUP} and ${TRAINING_LOG_GROUP} for training and simulation logs."
    # redirect stderr to stdout and have error messages sent to the same file as standard output
    roslaunch deepracer_simulation_environment $SIMULATION_LAUNCH_FILE publish_to_kinesis_stream:=false
fi