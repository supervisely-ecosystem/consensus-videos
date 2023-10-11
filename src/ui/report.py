import time
import supervisely as sly
from supervisely import VideoAnnotation
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import (
    Container,
    Text,
    Card,
    Table,
    GridGallery,
    Timeline,
    Field,
)
from supervisely.app import DataJson
import src.globals as g
import src.utils as utils


gt_video_info = None
pred_video_info = None
gt_annotation = None
pred_annotation = None
differences = []
left_name = ""
right_name = ""

overall_score = Text("")
overall_stats = Card(title="Overall Score", content=overall_score)

obj_count_per_class_table_columns = [
    "NAME",
    "GT Objects",
    "Labeled Objects",
    "Recall(Matched objects)",
    "Precision",
    "F-measure",
]
obj_count_per_class_table = Table(columns=obj_count_per_class_table_columns)
obj_count_per_class_last = Text()
obj_count_per_class = Card(
    title="OBJECTS COUNT PER CLASS",
    content=Container(widgets=[obj_count_per_class_table, obj_count_per_class_last], gap=5),
)

geometry_quality_table_columns = ["NAME", "Pixel Accuracy", "IOU"]
geometry_quality_table = Table(columns=geometry_quality_table_columns)
geometry_quality_last = Text()
geometry_quality = Card(
    title="GEOMETRY QUALITY",
    content=Container(widgets=[geometry_quality_table, geometry_quality_last], gap=5),
)

tags_stat_table_columns = [
    "NAME",
    "GT Tags",
    "Labeled Tags",
    "Precision",
    "Recall",
    "F-measure",
]
tags_stat_table = Table(columns=tags_stat_table_columns)
tags_stat_last = Text()
tags_stat = Card(title="TAGS", content=Container(widgets=[tags_stat_table, tags_stat_last], gap=5))

report_per_image_table_columns = [
    "FRAME #",
    "Objects Score",
    "Objects Missing",
    "Objects False Positive",
    "Tags Score",
    "Tags Missing",
    "Tags False Positive",
    "Geometry Score",
    "Overall Score",
]
report_per_image_table = Table(columns=report_per_image_table_columns)
timeline_fp = Timeline(1, [[1, 1]], ["red"])
timeline_fn = Timeline(1, [[1, 1]], ["red"])


def report_to_dict(report):
    d = {}
    for metric in report:
        if metric["metric_name"] not in d:
            d[metric["metric_name"]] = {}
        if metric["gt_frame_n"] not in d[metric["metric_name"]]:
            d[metric["metric_name"]][metric["gt_frame_n"]] = {
                "pred_frame_n": metric["pred_frame_n"]
            }
        d[metric["metric_name"]][metric["gt_frame_n"]][
            (metric["class_gt"], metric["tag_name"])
        ] = metric["value"]
    return d


def show_images(frame_n):
    global report_per_image_images
    report_per_image_images.loading = True
    report_per_image_images.clean_up()
    global gt_video_info
    global pred_video_info
    global gt_annotation
    global pred_annotation
    global differences
    global left_name
    global right_name

    # gt image
    frame_np = g.api.video.frame.download_np(gt_video_info.id, frame_n)
    utils.save_img(frame_np, "gt.jpg")

    # pred image
    frame_np = g.api.video.frame.download_np(pred_video_info.id, frame_n)
    utils.save_img(frame_np, "pred.jpg")

    # gt annotation
    frame_shape = (gt_video_info.frame_height, gt_video_info.frame_width)
    labels = [
        sly.Label(geometry=fig.geometry, obj_class=fig.video_object.obj_class)
        for fig in gt_annotation.figures
        if fig.frame_index == frame_n
    ]
    gt_ann = sly.Annotation(frame_shape, labels=labels)

    # pred annotation
    frame_shape = (pred_video_info.frame_height, pred_video_info.frame_width)
    labels = [
        sly.Label(geometry=fig.geometry, obj_class=fig.video_object.obj_class)
        for fig in pred_annotation.figures
        if fig.frame_index == frame_n
    ]
    pred_ann = sly.Annotation(frame_shape, labels=labels)

    # diff annotation
    frame_shape = (gt_video_info.frame_height, gt_video_info.frame_width)
    try:
        labels = [
            sly.Label(
                differences[frame_n],
                sly.ObjClass("difference", sly.Bitmap, (255, 0, 0)),
            )
        ]
    except:
        labels = []
    diff_ann = sly.Annotation(
        img_size=frame_shape,
        labels=labels,
    )

    report_per_image_images.append(
        f"/static/gt.jpg?{time.time()}", gt_ann, title=left_name, column_index=0
    )
    report_per_image_images.append(
        f"/static/pred.jpg?{time.time()}",
        pred_ann,
        title=right_name,
        column_index=1,
    )
    report_per_image_images.append(
        f"/static/gt.jpg?{time.time()}", diff_ann, title="Difference", column_index=2
    )

    DataJson().send_changes()
    report_per_image_images.loading = False


@report_per_image_table.click
def report_per_image_table_clicked(datapoint):
    row = datapoint.row
    frame_n = row["FRAME #"]

    # set timeline pointer
    timeline_fp.set_pointer(frame_n - 1)
    timeline_fn.set_pointer(frame_n - 1)

    show_images(frame_n - 1)


report_per_image_images = GridGallery(3, enable_zoom=True, sync_views=True, fill_rectangle=False)
report_per_image = Card(
    title="REPORT PER FAME",
    description="Click on a row to see annotation differences",
    content=Container(
        widgets=[
            Container(
                widgets=[
                    Field(title="False Positive matches", content=timeline_fp),
                    Field(title="False Negative matches", content=timeline_fn),
                ],
                gap=0,
            ),
            report_per_image_table,
            Card(content=report_per_image_images, collapsable=True),
        ]
    ),
)

results = Container(
    widgets=[
        overall_stats,
        obj_count_per_class,
        geometry_quality,
        tags_stat,
        report_per_image,
    ],
    gap=10,
)
layout = results


def get_overall_score(report):
    try:
        return report["overall-score"][0][("", "")]
    except KeyError:
        return 0


def get_obj_count_per_class_row(report, class_name):
    metrics = {
        "num-objects-gt": 0,
        "num-objects-pred": 0,
        "matches-recall": 1,
        "matches-precision": 1,
        "matches-f1": 1,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][0][(class_name, "")]
        except KeyError:
            pass
    return [
        class_name,
        str(metrics["num-objects-gt"]),
        str(metrics["num-objects-pred"]),
        f'{int(metrics["matches-recall"]*metrics["num-objects-gt"])} of {metrics["num-objects-gt"]} ({round(metrics["matches-recall"]*100, 2)}%)',
        f'{int(metrics["matches-precision"]*metrics["num-objects-pred"])} of {metrics["num-objects-pred"]} ({round(metrics["matches-precision"]*100, 2)}%)',
        f'{round(metrics["matches-f1"]*100, 2)}%',
    ]


def get_average_f_measure_per_class(report):
    try:
        return report["matches-f1"][0][("", "")]
    except KeyError:
        return 1


def get_geometry_quality_row(report, class_name):
    metrics = {
        "pixel-accuracy": 1,
        "iou": 1,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][0][(class_name, "")]
        except KeyError:
            pass

    return [
        class_name,
        f'{round(metrics["pixel-accuracy"]*100, 2)}%',
        f'{round(metrics["iou"]*100, 2)}%',
    ]


def get_average_iou(report):
    try:
        return report["iou"][0][("", "")]
    except KeyError:
        return 1


def get_tags_stat_table_row(report, tag_name):
    metrics = {
        "tags-total-gt": 0,
        "tags-total-pred": 0,
        "tags-precision": 1,
        "tags-recall": 1,
        "tags-f1": 1,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][0][("", tag_name)]
        except KeyError:
            pass
    return [
        tag_name,
        metrics["tags-total-gt"],
        metrics["tags-total-pred"],
        f'{int(metrics["tags-precision"]*metrics["tags-total-pred"])} of {metrics["tags-total-pred"]} ({round(metrics["tags-precision"]*100, 2)}%)',
        f'{int(metrics["tags-recall"]*metrics["tags-total-gt"])} of {metrics["tags-total-gt"]} ({round(metrics["tags-recall"]*100, 2)}%)',
        f'{round(metrics["tags-f1"]*100, 2)}%',
    ]


def get_average_f_measure_per_tags(report):
    try:
        return report["tags-f1"][0][("", "")]
    except KeyError:
        return 1


def get_report_per_image_row(report, frame_n):
    metrics = {
        "matches-f1": 0.0,
        "matches-false-negative": 0,
        "matches-false-positive": 0,
        "tags-f1": 0.0,
        "tags-false-negative": 0,
        "tags-false-positive": 0,
        "iou": 0.0,
        "overall-score": 0.0,
    }
    for metric_name in metrics.keys():
        try:
            metrics[metric_name] = report[metric_name][frame_n][("", "")]
        except KeyError:
            pass
    return [
        frame_n,
        f'{round(metrics["matches-f1"]*100, 2)}%',
        metrics["matches-false-negative"],
        metrics["matches-false-positive"],
        f'{round(metrics["tags-f1"]*100, 2)}%',
        metrics["tags-false-negative"],
        metrics["tags-false-positive"],
        f'{round(metrics["iou"]*100, 2)}%',
        f'{round(metrics["overall-score"]*100, 2)}%',
    ]


def clean_up():
    report_per_image_images.clean_up()
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [],
        }
    )
    obj_count_per_class_last.set(
        text=f"<b>Objects score (average F-measure) {0.00}%</b>", status="text"
    )

    geometry_quality_table.read_json(
        {
            "columns": geometry_quality_table_columns,
            "data": [],
        }
    )
    geometry_quality_last.set(f"<b>Geometry score (average IoU) {0.00}%</b>", status="text")

    tags_stat_table.read_json(
        {
            "columns": tags_stat_table_columns,
            "data": [],
        }
    )
    tags_stat_last.set("<b>Tags score (average F-measure) 0%</b>", status="text")

    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [],
        }
    )

    overall_score.set("-", status="text")


def get_intervals_with_colors(report: dict):
    matches_fn = []
    matches_fp = []
    good_fn = []
    good_fp = []
    for frame_n in range(1, gt_video_info.frames_count + 1):
        if report["matches-false-negative"][frame_n][("", "")] != 0:
            matches_fn.append([frame_n, frame_n])
        else:
            good_fn.append([frame_n, frame_n])
        if report["matches-false-positive"][frame_n][("", "")] != 0:
            matches_fp.append([frame_n, frame_n])
        else:
            good_fp.append([frame_n, frame_n])
    matches_fn = utils.unite_ranges(matches_fn)
    good_fn = utils.unite_ranges(good_fn)
    matches_fp = utils.unite_ranges(matches_fp)
    good_fp = utils.unite_ranges(good_fp)
    return (
        [*matches_fn, *good_fn],
        [*["red" for _ in range(len(matches_fn))], *["white" for _ in range(len(good_fn))]],
    ), (
        [*matches_fp, *good_fp],
        [*["red" for _ in range(len(matches_fp))], *["white" for _ in range(len(good_fp))]],
    )


@sly.timeit
def render_report(
    report,
    gt_video: VideoInfo,
    pred_video: VideoInfo,
    gt_video_ann: VideoAnnotation,
    pred_video_ann: VideoAnnotation,
    diffs,
    classes,
    tags,
    first_name,
    second_name,
):
    results.loading = True
    report_per_image_images.clean_up()

    global gt_video_info
    global pred_video_info
    global gt_annotation
    global pred_annotation
    global differences
    global left_name
    global right_name

    gt_video_info = gt_video
    pred_video_info = pred_video
    gt_annotation = gt_video_ann
    pred_annotation = pred_video_ann
    differences = diffs
    left_name = first_name
    right_name = second_name

    report = report_to_dict(report)

    # overall score
    def get_score_text(score):
        if score > 0.66:
            return f'<span style="color: green"><h2>{round(score*100, 2)}</h2></span>'
        if score > 0.33:
            return f'<span style="color: orange"><h2>{round(score*100, 2)}</h2></span>'
        return f'<span style="color: red"><h2>{round(score*100, 2)}</h2></span>'

    overall_score.set(get_score_text(get_overall_score(report)), status="text")

    # obj count per class
    obj_count_per_class_table.read_json(
        {
            "columns": obj_count_per_class_table_columns,
            "data": [get_obj_count_per_class_row(report, cls_name) for cls_name in classes],
        }
    )
    obj_count_per_class_last.set(
        text=f"<b>Objects score (average F-measure) {round(get_average_f_measure_per_class(report)*100, 2)}%</b>",
        status="text",
    )

    # geometry quality
    geometry_quality_table.read_json(
        {
            "columns": geometry_quality_table_columns,
            "data": [get_geometry_quality_row(report, cls_name) for cls_name in classes],
        }
    )
    geometry_quality_last.set(
        f"<b>Geometry score (average IoU) {round(get_average_iou(report)*100, 2)}%</b>",
        status="text",
    )

    # tags
    tags_stat_table.read_json(
        {
            "columns": tags_stat_table_columns,
            "data": [get_tags_stat_table_row(report, tag_name) for tag_name in tags],
        }
    )
    tags_stat_last.set(
        f"<b>Tags score (average F-measure) {round(get_average_f_measure_per_tags(report)*100, 2)}%</b>",
        status="text",
    )

    # per image
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [
                get_report_per_image_row(report, frame_n + 1)
                for frame_n in range(gt_video.frames_count)
            ],
        }
    )

    # set timeline
    (intervals_fn, colors_fn), (intervals_fp, colors_fp) = get_intervals_with_colors(report)
    timeline_fp.set(gt_video.frames_count, intervals_fp, colors_fp)
    timeline_fn.set(gt_video.frames_count, intervals_fn, colors_fn)

    results.loading = False


@timeline_fp.segment_selected
def timeline_fp_segment_selected(segment):
    print(segment)
    print(timeline_fp.get_pointer())
    frame_n = timeline_fp.get_pointer()
    show_images(frame_n)
