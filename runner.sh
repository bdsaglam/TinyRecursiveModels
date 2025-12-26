#!/bin/bash
# LICENSE: https://maximerobeyns.com/fragments/job_queue

JOBS_FILE="jobs.txt"

while true; do
    # Find the line number of the first un-executed job
    LINE_NUM=$(grep -n -m 1 "^\[ \]" "$JOBS_FILE" | cut -d: -f1)

    # Check if there are any un-executed jobs left
    if [ -z "$LINE_NUM" ]; then
        break
    fi

    # Extract the command and append this runner's PID
    JOB_LINE=$(sed -n "${LINE_NUM}p" "$JOBS_FILE")
    JOB_COMMAND=$(echo "$JOB_LINE" | sed -e 's/^\[ \] //')
    EXECUTING_JOB_LINE="[-] $JOB_COMMAND [$$]"

    # Replace the line with the executing status
    sed -i "${LINE_NUM}s#.*#$EXECUTING_JOB_LINE#" "$JOBS_FILE"

    # Execute the command
    eval "$JOB_COMMAND"
    STATUS=$?

    # Update the job status based on execution result
    if [ $STATUS -eq 0 ]; then
        # Update the job status to completed
        sed -i "/$$/s#.*#[x] $JOB_COMMAND#" "$JOBS_FILE"
    else
        # Update the job status to failed
        sed -i "/$$/s#.*#[!] $JOB_COMMAND#" "$JOBS_FILE"
    fi
done

echo "All jobs are completed."