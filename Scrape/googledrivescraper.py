from apiclient import discovery
from httplib2 import Http
import oauth2client
from oauth2client import file, client, tools
obj = lambda: None
lmao = {"auth_host_name":'localhost', 'noauth_local_webserver':'store_true', 'auth_host_port':[8080, 8090], 'logging_level':'ERROR'}
for k, v in lmao.items():
    setattr(obj, k, v)
    
# authorization boilerplate code
SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
store = file.Storage('token.json')
creds = store.get()
# The following will give you a link if token.json does not exist, the link allows the user to give this app permission
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('C:\\Users\\Rafid Mirza\\Documents\\New Folder\\client_id.json', SCOPES)
    creds = tools.run_flow(flow, store, obj)

import io
from googleapiclient.http import MediaIoBaseDownload
DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))
# if you get the shareable link, the link contains this id, replace the file_id below
file_id = '1uNE8MwrulwuGcg5vY_UgrG9n9EAWes12'
request = DRIVE.files().get_media(fileId=file_id)
# replace the filename and extension in the first field below
fh = io.FileIO('scrape.xml', mode='w')
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print("Download %d%%." % int(status.progress() * 100))

# importing os module  
import os  
  
# importing shutil module  
import shutil  
  
# Source path  
source = 'C:/Users/Rafid Mirza/scrape.xml'
  
# Destination path  
destination = 'C:/Users/Rafid Mirza/Documents/New Folder'
  
# Move the content of  
# source to destination  
dest = shutil.move(source, destination)  