#!/bin/bash

. common.sh

Scripts/$protocol.sh $prog $run_opt

# allow debugging with docker
exit 0
