#!/usr/bin/env python3

import os
import argparse
import subprocess
from pathlib import Path
from loguru import logger

def scp_to_remote(source_path, remote_hosts, remote_user="your_username", remote_base_dir= "/home/tairanhe/workspace/RoboVerse/roboverse/logs"):
    """
    Copy checkpoint file to the same directory structure on remote machines
    
    Args:
        source_path (str): Path to source file
        remote_hosts (list): List of remote hostnames/IPs
        remote_user (str): Remote username for SSH
    """

    source_path = Path(source_path)
    relative_path = Path(remote_base_dir) / source_path.parent
    
    # Ensure source file exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    
    for host in remote_hosts:
        logger.info(f"Copying to {host}...")
        
        # Create remote directory structure
        ssh_mkdir_cmd = [
            "ssh",
            f"{remote_user}@{host}",
            f"mkdir -p {relative_path}"
        ]
        
        try:
            subprocess.run(ssh_mkdir_cmd, check=True)
            
            # SCP the file
            scp_cmd = [
                "scp",
                str(source_path),
                f"{remote_user}@{host}:{relative_path}"
            ]
            subprocess.run(scp_cmd, check=True)
            print(f"Successfully copied to {host}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error copying to {host}: {e}")

def main():
    parser = argparse.ArgumentParser(description="SCP checkpoint files to remote machines")
    parser.add_argument("--hosts", nargs="+", default=["172.26.56.3"],
                        help="List of remote hosts (default: your.default.host)")
    parser.add_argument("--user", default="tairanhe", help="Remote username")
    parser.add_argument("--remote-dir", default="~/workspace/RoboVerse/",
                        help="Remote base directory path")
    parser.add_argument("--source", help="Path to the source file")
    args = parser.parse_args()

    
    scp_to_remote(args.source, args.hosts, args.user, args.remote_dir)

if __name__ == "__main__":
    main()