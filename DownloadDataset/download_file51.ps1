
            # Define the directory where files will be downloaded
                $downloadDir = "kitti_data"
                
                # Create the directory if it doesn't exist
                if (-not (Test-Path -Path $downloadDir)) {
                    New-Item -ItemType Directory -Path $downloadDir
                }
                
                # URL to download
                $url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip"
                
                # Download the file
                $fileName = Split-Path $url -Leaf
                Invoke-WebRequest -Uri $url -OutFile "$downloadDir\$fileName"
            