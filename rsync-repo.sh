#!/usr/bin/env bash

set -e

function check_args_num() {
  num_args=$1
  shift 1

  if [[ "$#" -gt "$num_args" ]]; then
    echo "Illegal number of parameters for the command has been passed. Number of arguments should be less or equal to "$num_args
    exit 1
  fi
}

function check_host() {
    if [[ -z "${SYNC_HOST+x}" ]]; then
        echo "SYNC_HOST is not defined"
        exit 1
    fi
}

function get_dir() {
    if [[ -z "${SYNC_DIR+x}" ]]; then
        sync_dir=${PWD##*/}
    else
        sync_dir=$SYNC_DIR
    fi

    echo $sync_dir
}

# Be aware about '-u' option of rsync.
# it forces rsync not to copy files which are newer on the contrary side
function sync() {
    arg=$2

    if [[ "$arg" = "-f" ]] && [[ "$1" = true ]] ; then
         rsyncargs="-arv --delete"
    else
         rsyncargs="-aurv"
    fi

    check_host
    host=$SYNC_HOST

    sync_dir=$(get_dir)

    if [[ "$1" = true ]] ; then
        from="."
        to="${host}:${sync_dir}"
    else
        from="${host}:${sync_dir}/*"
        to="."
    fi

    echo "Syncing from $from to $to"

    rsync $rsyncargs \
        --exclude 'kafka_2.11-*' \
        --exclude '*__pycache__*' \
        --exclude '*target*' \
	--exclude '*ipynb_checkpints' \
        --exclude 'venv' \
        --exclude 'download-cache' \
        --exclude '*.iml' \
        --exclude '.idea' \
        --exclude 'spark/work' \
        --exclude '*/.git/*' \
        --exclude '.git/*' \
        --exclude '*.log' \
        --exclude '*/.idea/*' \
        --exclude 'work' \
        --exclude '*.hdf5' \
        --exclude '*.jpg' \
        --exclude '*.png' \
        --exclude 'examples/tmp' \
        --exclude '*.rpm' \
        --exclude '*/.vagrant' \
        --exclude '*.vdi' \
        "$from" $to
}

function help () {
    echo "This script synchronizes remote and current folder in both
directions using rsync.

    upload [-f] - upload data to remote folder from the local one.
        -f use '-arv --delete' for rsync instead of '-aurv'. Updates and new files on remote side will be deleted.

    download - download data from remote folder to the local one

you should set SYNC_HOST environment variable, for example:
    export SYNC_HOST=192.168.13.110

you may set also SYNC_DIR environment variable, for example:
    export SYNC_DIR=Notebooks/MyOwnDirectory"
}

function main () {
    case "$1" in
    "upload")
        check_args_num 2 "${@}"
        shift 1
        sync true "${@}"
        ;;
    "download")
        check_args_num 1 "${@}"
        sync false
        ;;
    "help")
        check_args_num 1 "${@}"
        help
        ;;
    *)
        echo "Unknown command: "$1
        ;;
    esac
}

main "${@}"
