docker run -d --name verl_rjy \
  --gpus all \
  --shm-size=32g \
  -p 8265:8265 \
  -v /home/rjy/projects/verl:/workspace \
  -v /data/common:/data/common:ro \
  -v /data/rjy:/data/rjy \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  verl-0.7:latest \
  sleep infinity

# bash verl/experimental/one_step_off_policy/shell/run_dynamic_resize_controller.sh &> 1.log