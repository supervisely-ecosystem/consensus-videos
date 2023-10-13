from pathlib import Path
from typing import List
from supervisely import (
    VideoAnnotation,
    KeyIdMap,
    Frame,
    VideoObjectCollection,
    VideoTagCollection,
    FrameCollection,
)
from supervisely.api.video.video_api import VideoInfo
from supervisely.imaging.color import rgb2hex
from supervisely.decorators.profile import timeit
from supervisely.imaging.image import write as write_image

import src.globals as g


video_annotators = {}
video_anns = {}
key_id_maps = {}
ds_video_infos = {}


def wrap_in_tag(text, color):
    return f'<i class="zmdi zmdi-circle" style="margin-right: 3px; color: {rgb2hex(color)}"></i><span style="margin-right: 6px;">{text}</span>'


def get_project_meta(project_id):
    if project_id not in g.project_metas:
        g.project_metas[project_id] = g.api.project.get_meta(project_id)
    return g.project_metas[project_id]


def get_video(video_id) -> VideoInfo:
    if video_id not in g.all_videos:
        g.all_videos[video_id] = g.api.video.get_info_by_id(video_id)
    return g.all_videos[video_id]


@timeit
def get_videos(dataset_ids: List[int]):
    all_videos = []
    for dataset_id in dataset_ids:
        if dataset_id not in ds_video_infos:
            ds_video_infos[dataset_id] = g.api.video.get_list(dataset_id)
            g.all_videos.update({video.id: video for video in ds_video_infos[dataset_id]})
        all_videos.extend(ds_video_infos[dataset_id])
    return all_videos


def get_video_ann(video_id):
    if video_id not in video_anns:
        key_id_map = KeyIdMap()
        video_info = get_video(video_id)
        project_meta = get_project_meta(video_info.project_id)
        ann_json = g.api.video.annotation.download(video_id)
        video_anns[video_id] = VideoAnnotation.from_json(ann_json, project_meta, key_id_map)
    return video_anns[video_id]


@timeit
def get_annotators(videos_ids: List[int]):
    annotators = set()
    key_id_map = g.key_id_map
    for video_id in videos_ids:
        if video_id not in video_annotators:
            video_annotators[video_id] = set()
            video_info = g.api.video.get_info_by_id(video_id)
            ann_info = g.api.video.annotation.download(video_info.id)
            ann = VideoAnnotation.from_json(
                ann_info, g.project_metas[video_info.project_id], key_id_map
            )
            for obj in ann.objects:
                video_annotators[video_id].add(obj.labeler_login)
        annotators = annotators.union(video_annotators[video_id])
    return annotators


@timeit
def get_score(report: List[dict]):
    if "error" in report:
        return "Error"
    for metric in report:
        if metric["metric_name"] == "overall-score":
            if metric["class_gt"] == "" and metric["tag_name"] == "" and metric["gt_frame_n"] == 0:
                return metric["value"]
    return 0


@timeit
def get_common_images(first_img_infos, second_img_infos):
    second_img_name_to_idx = {img.name: i for i, img in enumerate(second_img_infos)}
    paired_infos = []
    for first_img in first_img_infos:
        if first_img.name in second_img_name_to_idx:
            paired_infos.append(
                (first_img, second_img_infos[second_img_name_to_idx[first_img.name]])
            )
    return [paired_info[0] for paired_info in paired_infos], [
        paired_info[1] for paired_info in paired_infos
    ]


@timeit
def filter_objects_by_user(
    first_video_ann: VideoAnnotation,
    second_video_ann: VideoAnnotation,
    first_login,
    second_login,
):
    first_filtered_objects = VideoObjectCollection(
        [obj for obj in first_video_ann.objects if obj.labeler_login == first_login]
    )
    first_filtered_frames = FrameCollection()
    for frame in first_video_ann.frames:
        frame: Frame
        figures = [fig for fig in frame.figures if fig.labeler_login == first_login]
        first_filtered_frames = first_filtered_frames.add(
            frame.clone(index=frame.index, figures=figures)
        )
    first_filtered_tags = VideoTagCollection(
        [tag for tag in first_video_ann.tags if tag.labeler_login == first_login]
    )
    first_ann_info = first_video_ann.clone(
        objects=first_filtered_objects, frames=first_filtered_frames, tags=first_filtered_tags
    )

    second_filtered_objects = VideoObjectCollection(
        [obj for obj in second_video_ann.objects if obj.labeler_login == second_login]
    )
    second_filtered_frames = FrameCollection()
    for frame in second_video_ann.frames:
        frame: Frame
        second_filtered_frames = second_filtered_frames.add(
            frame.clone(figures=[fig for fig in frame.figures if fig.labeler_login == second_login])
        )
    second_filtered_tags = VideoTagCollection(
        [tag for tag in second_video_ann.tags if tag.labeler_login == second_login]
    )
    second_ann_info = second_video_ann.clone(
        objects=second_filtered_objects, frames=second_filtered_frames, tags=second_filtered_tags
    )

    return first_ann_info, second_ann_info


@timeit
def get_classes(video_ann: VideoAnnotation):
    classes = set()
    for fig in video_ann.figures:
        classes.add(fig.video_object.obj_class.name)
    return classes


@timeit
def get_class_matches(first_classes, second_classes):
    return {class_name: class_name for class_name in first_classes.intersection(second_classes)}


@timeit
def get_tags_whitelists(first_video_ann: VideoAnnotation, second_video_ann: VideoAnnotation):
    first_tag_whitelist = set()
    first_obj_tags_whitelist = set()
    for tag in first_video_ann.tags:
        first_tag_whitelist.add(tag.name)
    for obj in first_video_ann.objects:
        for tag in obj.tags:
            first_obj_tags_whitelist.add(tag.name)
    second_tag_whitelist = set()
    second_obj_tags_whitelist = set()
    for tag in second_video_ann.tags:
        second_tag_whitelist.add(tag.name)
    for obj in second_video_ann.objects:
        for tag in obj.tags:
            second_obj_tags_whitelist.add(tag.name)

    tags_whitelist = list(first_tag_whitelist.intersection(second_tag_whitelist))
    obj_tags_whitelist = list(first_obj_tags_whitelist.intersection(second_obj_tags_whitelist))
    return tags_whitelist, obj_tags_whitelist


def get_project_by_id(project_id):
    if project_id not in g.data["projects"]:
        g.data["projects"][project_id] = g.api.project.get_info_by_id(project_id)
    return g.data["projects"][project_id]


def get_project_by_name(workspace_id, project_name):
    for project in g.data["projects"].values():
        if project.name == project_name and project.workspace_id == workspace_id:
            return project
    project = g.api.project.get_info_by_name(workspace_id, project_name)
    if project is not None:
        g.data["projects"][project.id] = project
    return project


def get_dataset_by_id(dataset_id):
    if dataset_id not in g.data["datasets"]:
        g.data["datasets"][dataset_id] = g.api.dataset.get_info_by_id(dataset_id)
    return g.data["datasets"][dataset_id]


def get_dataset_by_name(project_id, dataset_name):
    for dataset in g.data["datasets"]:
        if dataset.name == dataset_name and dataset.project_id == project_id:
            return dataset
    dataset = g.api.dataset.get_info_by_name(project_id, dataset_name)
    if dataset is not None:
        g.data["datasets"][dataset.id] = dataset
    return dataset


@timeit
def save_img(np, filename):
    write_image(str(Path(g.TEMP_DATA_PATH).joinpath(filename)), np)


def unite_ranges(ranges: List[List[int]]):
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    res = [ranges[0]]
    for i in range(1, len(ranges)):
        if ranges[i][0] <= res[-1][1] + 1:
            res[-1][1] = max(res[-1][1], ranges[i][1])
        else:
            res.append(ranges[i])
    return res


@timeit
def download_frame(video_id, frame_n):
    return g.api.video.frame.download_np(video_id, frame_n)


@timeit
def download_video(video_id, filename):
    g.api.video.download_path(video_id, str(Path(g.TEMP_DATA_PATH).joinpath(filename)))
