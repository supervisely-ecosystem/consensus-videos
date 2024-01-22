import os
from dotenv import load_dotenv
import supervisely as sly


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

SERVER_ADDRESS = sly.env.server_address()
API_TOKEN = sly.env.api_token()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
PROJECT_ID = sly.env.project_id(raise_not_found=False)
DATASET_ID = sly.env.dataset_id(raise_not_found=False)
TEMP_DATA_PATH = "/tmp/consensus-videos"
if sly.is_development():
    TEMP_DATA_PATH = "temp_data"
if not os.path.exists(TEMP_DATA_PATH):
    os.makedirs(TEMP_DATA_PATH)
key_id_map = sly.KeyIdMap()

api = sly.Api()
workspace = api.workspace.get_info_by_id(WORKSPACE_ID)
team = api.team.get_info_by_id(TEAM_ID)
all_users = {user.id: user for user in api.user.get_team_members(TEAM_ID)}
all_projects = None
all_datasets = None
all_videos = None

if PROJECT_ID:
    all_projects = {PROJECT_ID: api.project.get_info_by_id(PROJECT_ID)}
else:
    all_projects = {
        project.id: project
        for project in api.project.get_list(WORKSPACE_ID)
        if project.type == str(sly.ProjectType.VIDEOS)
    }

if DATASET_ID:
    all_datasets = {DATASET_ID: api.dataset.get_info_by_id(DATASET_ID)}
else:
    all_datasets = {
        dataset.id: dataset
        for project in all_projects.values()
        for dataset in api.dataset.get_list(project.id)
    }

project_metas = {
    project.id: sly.ProjectMeta.from_json(api.project.get_meta(project.id))
    for project in all_projects.values()
}

data = {"projects": {}, "datasets": {}}
all_videos = {}
