# utils/google_drive.py
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import io
import pandas as pd
from urllib.parse import parse_qs, urlparse

# Define the OAuth 2.0 scopes needed for Drive access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Use localhost port that Streamlit uses
REDIRECT_URI = 'http://localhost:8501'

def check_for_auth_code():
    """
    Check the URL query parameters for an authorization code.
    Returns the code if found, None otherwise.
    """
    try:
        # Get the query parameters from the URL
        query_params = st.experimental_get_query_params()
        
        # Check if 'code' is in the parameters
        if 'code' in query_params:
            return query_params['code'][0]
    except:
        pass
    
    return None

def get_credentials(key_prefix=""):
    """
    Get and validate Google Drive credentials, handling authentication if needed.
    Handles redirect flow automatically.
    
    Args:
        key_prefix: Prefix for making the authentication key unique
    """
    creds = None
    
    # Check if we already have credentials in session state
    if 'google_creds' in st.session_state:
        creds = Credentials.from_authorized_user_info(st.session_state.google_creds)
    
    # If credentials are valid, return them
    if creds and creds.valid:
        return creds
        
    # If credentials are expired but can be refreshed, do so
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        st.session_state.google_creds = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        return creds
    
    # Check for an authorization code in the URL (from redirect)
    auth_code = check_for_auth_code()
    if auth_code:
        try:
            # Set up the flow for token exchange
            if os.path.exists('credentials.json'):
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', 
                    SCOPES,
                    redirect_uri=REDIRECT_URI
                )
            else:
                # For deployment, use environment variables
                client_config = {
                    "web": {
                        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                        "redirect_uris": [REDIRECT_URI]
                    }
                }
                flow = InstalledAppFlow.from_client_config(
                    client_config, 
                    SCOPES,
                    redirect_uri=REDIRECT_URI
                )
                
            # Exchange the code for credentials
            flow.fetch_token(code=auth_code)
            creds = flow.credentials
            
            # Store credentials in session state
            st.session_state.google_creds = {
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }
            
            st.success("Autenticação com Google Drive concluída com sucesso!")
            # Clear URL parameters to avoid reprocessing the code on refresh
            st.experimental_set_query_params()
            return creds
            
        except Exception as e:
            st.error(f"Falha na autenticação: {str(e)}")
            # Clear URL parameters to avoid reprocessing the code on refresh
            st.experimental_set_query_params()
            return None
    
    # If not authenticated, display login button
    st.write("Para acessar seus arquivos no Google Drive, autentique-se primeiro:")
    
    # Set up the flow for authorization URL generation
    if os.path.exists('credentials.json'):
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', 
            SCOPES,
            redirect_uri=REDIRECT_URI
        )
    else:
        # For deployment, use environment variables
        client_config = {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uris": [REDIRECT_URI]
            }
        }
        flow = InstalledAppFlow.from_client_config(
            client_config, 
            SCOPES,
            redirect_uri=REDIRECT_URI
        )
    
    # Generate the authorization URL
    auth_url, _ = flow.authorization_url(
        access_type='offline', 
        include_granted_scopes='true',
        prompt='consent'
    )
    
    # Display the authentication button
    auth_button_key = f"{key_prefix}_auth_button" if key_prefix else "main_auth_button"
    if st.button("Autenticar com Google Drive", key=auth_button_key):
        # Open the authorization URL in a new tab
        st.markdown(f'<meta http-equiv="refresh" content="0;URL=\'{auth_url}\'" />', unsafe_allow_html=True)
        st.markdown(f"[Clique aqui se não for redirecionado automaticamente]({auth_url})")
    
    return None

def get_drive_service(key_prefix=""):
    """
    Get authenticated Google Drive service object.
    First ensure authentication is complete.
    
    Args:
        key_prefix: Prefix for making the authentication key unique
        
    Returns:
        Google Drive service object or None if not authenticated
    """
    # Get credentials, which handles the authentication flow
    creds = get_credentials(key_prefix)
    
    if creds:
        # Build and return the Drive service
        return build('drive', 'v3', credentials=creds)
    
    return None

def get_shared_folder_id():
    """
    Get the ID of the shared folder from environment variables or configuration.
    """
    folder_id = os.getenv("GOOGLE_DRIVE_SHARED_FOLDER_ID")
    return folder_id

def list_files_in_folder(service, folder_id=None, file_type=None, query_text=None):
    """
    List files in a Google Drive folder with optional filtering.
    
    Args:
        service: Google Drive service object
        folder_id: ID of the folder to list files from (None for root/My Drive)
        file_type: Optional file type filter (e.g., 'csv')
        query_text: Optional text to search in file names
        
    Returns:
        List of file metadata dictionaries
    """
    query_parts = []
    
    # Filter by parent folder
    if folder_id:
        query_parts.append(f"'{folder_id}' in parents")
    
    # Filter by file type
    if file_type:
        if file_type.lower() == 'csv':
            # For CSVs, need to check the file extension since Drive doesn't have a specific MIME type
            query_parts.append("name contains '.csv'")
    
    # Filter by name if search text provided
    if query_text:
        query_parts.append(f"name contains '{query_text}'")
    
    # Only show non-trashed files
    query_parts.append("trashed = false")
    
    # Combine all query parts
    query = " and ".join(query_parts)
    
    # Execute the query
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, webViewLink)",
        pageSize=50
    ).execute()
    
    return results.get('files', [])

def list_folders(service, parent_id=None):
    """
    List folders in a Google Drive location.
    
    Args:
        service: Google Drive service object
        parent_id: ID of the parent folder (None for root/My Drive)
        
    Returns:
        List of folder metadata dictionaries
    """
    query = "mimeType='application/vnd.google-apps.folder' and trashed = false"
    
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query,
        fields="files(id, name)",
        pageSize=50
    ).execute()
    
    return results.get('files', [])

def download_file(service, file_id):
    """
    Download a file from Google Drive.
    
    Args:
        service: Google Drive service object
        file_id: ID of the file to download
        
    Returns:
        BytesIO object containing file content
    """
    request = service.files().get_media(fileId=file_id)
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)
    
    done = False
    with st.spinner('Baixando arquivo...'):
        while not done:
            status, done = downloader.next_chunk()
            st.progress(int(status.progress() * 100))
    
    file_content.seek(0)
    return file_content

def file_selector_ui(key_prefix=""):
    """
    UI component for selecting files from Google Drive.
    
    Args:
        key_prefix: Prefix for unique session state keys
        
    Returns:
        DataFrame if a CSV is selected and loaded, otherwise None
    """
    service = get_drive_service(key_prefix)
    
    if not service:
        return None
    
    # Initialize navigation state if not already done
    nav_key = f"{key_prefix}_drive_nav"
    if nav_key not in st.session_state:
        # Try to start with the shared folder if available
        shared_folder_id = get_shared_folder_id()
        if shared_folder_id:
            try:
                # Get the folder name for better UI
                folder_info = service.files().get(fileId=shared_folder_id, fields="name").execute()
                folder_name = folder_info.get('name', 'Shared Folder')
                st.session_state[nav_key] = [
                    {'name': 'Root', 'id': None},
                    {'name': folder_name, 'id': shared_folder_id}
                ]
            except:
                st.session_state[nav_key] = [
                    {'name': 'Root', 'id': None}
                ]
        else:
            st.session_state[nav_key] = [
                {'name': 'Root', 'id': None}
            ]
    
    # Get current folder from navigation path
    current_folder = st.session_state[nav_key][-1]
    current_folder_id = current_folder['id']
    
    # Search box
    search_query = st.text_input("Buscar arquivos:", key=f"{key_prefix}_search")
    
    # Show breadcrumb navigation
    st.write("Localização:")
    breadcrumb_cols = st.columns(min(5, len(st.session_state[nav_key])))
    for i, folder in enumerate(st.session_state[nav_key][-5:] if len(st.session_state[nav_key]) > 5 else st.session_state[nav_key]):
        col_idx = i if len(st.session_state[nav_key]) <= 5 else i - (len(st.session_state[nav_key]) - 5)
        with breadcrumb_cols[col_idx]:
            if st.button(folder['name'], key=f"{key_prefix}_nav_{i}"):
                # Navigate to this level in the breadcrumb
                st.session_state[nav_key] = st.session_state[nav_key][:i+1]
                st.experimental_rerun()
    
    # List folders
    folders = list_folders(service, current_folder_id)
    if search_query:
        folders = [f for f in folders if search_query.lower() in f['name'].lower()]
    
    if folders:
        st.write("Pastas:")
        folder_cols = st.columns(3)
        for i, folder in enumerate(folders):
            with folder_cols[i % 3]:
                if st.button(f"📁 {folder['name']}", key=f"{key_prefix}_folder_{i}"):
                    # Navigate into this folder
                    st.session_state[nav_key].append({
                        'name': folder['name'],
                        'id': folder['id']
                    })
                    st.experimental_rerun()
    
    # List CSV files
    files = list_files_in_folder(service, current_folder_id, 'csv', search_query)
    
    if files:
        st.write("Arquivos CSV:")
        file_cols = st.columns(2)
        for i, file in enumerate(files):
            with file_cols[i % 2]:
                if st.button(f"📄 {file['name']}", key=f"{key_prefix}_file_{i}"):
                    try:
                        with st.spinner(f"Baixando {file['name']}..."):
                            # Download the file
                            file_content = download_file(service, file['id'])
                            
                            # Load as DataFrame
                            df = pd.read_csv(file_content)
                            
                            # Store in session state
                            st.session_state[f"{key_prefix}_selected_file"] = {
                                'name': file['name'],
                                'id': file['id'],
                                'data': df
                            }
                            
                            return df
                    except Exception as e:
                        st.error(f"Erro ao baixar ou processar o arquivo: {str(e)}")
                        return None
    else:
        st.info("Nenhum arquivo CSV encontrado nesta pasta")
    
    return None

def select_csv_from_drive(label, key):
    """
    Main function to select and load a CSV from Google Drive.
    
    Args:
        label: Text label for the file selection
        key: Unique key for this component
        
    Returns:
        DataFrame if a file is selected and loaded, otherwise None
    """
    st.write(f"### {label}")
    
    # Check if we already have a selected file
    if f"{key}_selected_file" in st.session_state:
        file_info = st.session_state[f"{key}_selected_file"]
        st.success(f"Arquivo selecionado: {file_info['name']}")
        
        if st.button("Selecionar outro arquivo", key=f"{key}_change"):
            # Clear the selection and show file browser
            del st.session_state[f"{key}_selected_file"]
            st.experimental_rerun()
        
        return file_info['data']
    
    # Otherwise show the file browser
    return file_selector_ui(key)