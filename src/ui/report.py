import copy
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
    InputNumber,
    Button,
    Switch,
    Flexbox,
)
from supervisely.app import DataJson
import src.utils as utils


GOOD_TIMELINE_COLOR = "#91B974"
ERROR_TIMELINE_COLOR = "#FF6458"
EMPTY_TIMELINE_COLOR = "#d1dbe5"


gt_video_info = None
pred_video_info = None
gt_annotation = None
pred_annotation = None
differences = []
left_name = ""
right_name = ""
current_report = None
current_frame_range = [1, 1]

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
report_per_image_table.click(lambda x: x)
timeline = Timeline(1, [], [])
timeline_select_frame = InputNumber(min=1, max=1, value=1)
timeline_filters_widgets = {
    "objects_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
    "objects_fp": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "objects_fn": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "tags_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
    "tags_fp": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "tags_fn": {"switch": Switch(), "input": InputNumber(min=0, value=0)},
    "geometry_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
    "overall_score": {
        "switch": Switch(),
        "input": InputNumber(min=0, max=100, value=80, step=0.1, precision=1),
    },
}
timeline_filters_apply_btn = Button("Apply")
timeline_filters = Flexbox(
    widgets=[
        Container(
            [
                Field(
                    title="Objects score",
                    description="If frame Objects Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["objects_score"]["switch"],
                            timeline_filters_widgets["objects_score"]["input"],
                        ],
                    ),
                ),
                Field(
                    title="Objects FN",
                    description="If Missing Objects number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["objects_fn"]["switch"],
                            timeline_filters_widgets["objects_fn"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Objects FP",
                    description="If False Positive Objects number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["objects_fp"]["switch"],
                            timeline_filters_widgets["objects_fp"]["input"],
                        ]
                    ),
                ),
            ]
        ),
        Container(
            [
                Field(
                    title="Tags score",
                    description="If frame Tags Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["tags_score"]["switch"],
                            timeline_filters_widgets["tags_score"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Tags FP",
                    description="If Missing Tags number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["tags_fp"]["switch"],
                            timeline_filters_widgets["tags_fp"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Tags FN",
                    description="If False Positive Tags number is more than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["tags_fn"]["switch"],
                            timeline_filters_widgets["tags_fn"]["input"],
                        ]
                    ),
                ),
            ]
        ),
        Container(
            [
                Field(
                    title="Geometry score",
                    description="If frame Geometry Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["geometry_score"]["switch"],
                            timeline_filters_widgets["geometry_score"]["input"],
                        ]
                    ),
                ),
                Field(
                    title="Overall score",
                    description="If frame Overall Score is less than this value, it will be marked as error",
                    content=Flexbox(
                        widgets=[
                            timeline_filters_widgets["overall_score"]["switch"],
                            timeline_filters_widgets["overall_score"]["input"],
                        ]
                    ),
                ),
            ]
        ),
    ],
    gap=30,
)


@timeline_select_frame.value_changed
def timeline_select_frame_cb(frame_n):
    global current_frame_range
    pointer = frame_n - current_frame_range[0]
    timeline.set_pointer(pointer)
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [get_report_per_image_row(report_to_dict(current_report), frame_n)],
        }
    )
    show_images(frame_n)


def get_timeline_filters():
    filters = {}
    for metric_name in [
        "objects_score",
        "objects_fp",
        "objects_fn",
        "tags_score",
        "tags_fp",
        "tags_fn",
        "geometry_score",
        "overall_score",
    ]:
        if timeline_filters_widgets[metric_name]["switch"].is_switched():
            filters[metric_name] = timeline_filters_widgets[metric_name]["input"].get_value()
        else:
            filters[metric_name] = None
    return filters


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
    global gt_video_info
    global pred_video_info
    global gt_annotation
    global pred_annotation
    global differences
    global left_name
    global right_name

    report_per_image_images.loading = True
    report_per_image_images.clean_up()

    # gt image
    frame_np = utils.download_frame(gt_video_info.id, frame_n - 1)
    utils.save_img(frame_np, "gt.jpg")

    # pred image
    if pred_video_info.id != gt_video_info.id:
        frame_np = utils.download_frame(pred_video_info.id, frame_n - 1)
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
                differences[frame_n - current_frame_range[0]],
                sly.ObjClass("difference", sly.Bitmap, (255, 0, 0)),
            )
        ]
    except:
        labels = []
    diff_ann = sly.Annotation(
        img_size=frame_shape,
        labels=labels,
    )
    report_per_image_images.clean_up()
    report_per_image_images.append(
        f"./static/gt.jpg?{time.time()}", gt_ann, title=left_name, column_index=0
    )
    report_per_image_images.append(
        f"./static/pred.jpg?{time.time()}",
        pred_ann,
        title=right_name,
        column_index=1,
    )
    report_per_image_images.append(
        f"./static/gt.jpg?{time.time()}", diff_ann, title="Difference", column_index=2
    )

    DataJson().send_changes()
    report_per_image_images.loading = False


report_per_image_images = GridGallery(3, enable_zoom=True, sync_views=True, fill_rectangle=False)
report_per_image = Card(
    title="REPORT PER FAME",
    description="Set filters to see frames with errors and click on the timeline to see the difference",
    content=Container(
        widgets=[
            Field(
                title="Timeline filters",
                description="Select metrics thresholds to display error frames on the timeline. Grey means there are no annotations on both frames for selected metrics",
                content=timeline_filters,
            ),
            Field(
                title="Timeline",
                content=Container(
                    widgets=[
                        timeline,
                        Text("<span>Frame:</span>"),
                        timeline_select_frame,
                    ],
                    direction="horizontal",
                    fractions=[1, 0, 0],
                    style="place-items: center;",
                ),
            ),
            report_per_image_table,
            report_per_image_images,
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


def get_report_per_image_row_values(report, frame_n):
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
        metrics["matches-f1"] * 100,
        metrics["matches-false-negative"],
        metrics["matches-false-positive"],
        metrics["tags-f1"] * 100,
        metrics["tags-false-negative"],
        metrics["tags-false-positive"],
        metrics["iou"] * 100,
        metrics["overall-score"] * 100,
    ]


def get_report_per_image_row(report, frame_n):
    values = get_report_per_image_row_values(report, frame_n)
    return [
        frame_n,
        f"{round(values[0], 2)}%",
        values[1],
        values[2],
        f"{round(values[3], 2)}%",
        values[4],
        values[5],
        f"{round(values[6], 2)}%",
        f"{round(values[7], 2)}%",
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
            "data": [["" for _ in report_per_image_table_columns]],
        }
    )

    overall_score.set("-", status="text")


def get_intervals_with_colors(report: dict, filters: dict = None, frame_range=[1, 1]):
    def get_color(frame_n):
        values = get_report_per_image_row_values(report, frame_n)
        current_color = EMPTY_TIMELINE_COLOR
        if any(
            filters[filter_name] is not None
            for filter_name in ("objects_score", "objects_fn", "objects_fp")
        ):
            if any(values[0:3]):
                if filters["objects_score"] is not None and values[0] < filters["objects_score"]:
                    return ERROR_TIMELINE_COLOR
                if filters["objects_fn"] is not None and values[1] > filters["objects_fn"]:
                    return ERROR_TIMELINE_COLOR
                if filters["objects_fp"] is not None and values[2] > filters["objects_fp"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR
        if any(
            filters[filter_name] is not None for filter_name in ("tags_score", "tags_fn", "tags_fp")
        ):
            if any(values[3:6]):
                if filters["tags_score"] is not None and values[3] < filters["tags_score"]:
                    return ERROR_TIMELINE_COLOR
                if filters["tags_fn"] is not None and values[4] > filters["tags_fn"]:
                    return ERROR_TIMELINE_COLOR
                if filters["tags_fp"] is not None and values[5] > filters["tags_fp"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR
        if filters["geometry_score"] is not None:
            if any(v != 0 for v in values[0:3]):
                if values[6] < filters["geometry_score"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR
        if filters["overall_score"] is not None:
            if any(v != 0 for v in values[0:7]):
                if values[7] < filters["overall_score"]:
                    return ERROR_TIMELINE_COLOR
                current_color = GOOD_TIMELINE_COLOR

        return current_color

    colors = {
        ERROR_TIMELINE_COLOR: [],
        GOOD_TIMELINE_COLOR: [],
        EMPTY_TIMELINE_COLOR: [],
    }
    for frame_n in range(frame_range[0], frame_range[1] + 1):
        color = get_color(frame_n)
        colors[color].append([frame_n - frame_range[0], frame_n - frame_range[0]])

    for color, ranges in colors.items():
        united_ranges = utils.unite_ranges(ranges)
        colors[color] = united_ranges
    return (
        [range for color, ranges in colors.items() for range in ranges],
        [color for color, ranges in colors.items() for range in ranges],
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
    frame_range,
):
    results.loading = True

    global gt_video_info
    global pred_video_info
    global gt_annotation
    global pred_annotation
    global differences
    global left_name
    global right_name
    global current_report
    global current_frame_range

    gt_video_info = gt_video
    pred_video_info = pred_video
    gt_annotation = gt_video_ann
    pred_annotation = pred_video_ann
    differences = diffs
    left_name = first_name
    right_name = second_name
    current_frame_range = frame_range

    current_report = copy.deepcopy(report)
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

    # set timeline
    timeline_select_frame.min = current_frame_range[0]
    timeline_select_frame.max = current_frame_range[1]
    timeline_select_frame.value = timeline_select_frame.min
    apply_timeline_filters()

    results.loading = False


@timeline.click
def timeline_click_cb(pointer):
    global current_report
    global report_per_image_table
    global current_frame_range
    frame_n = pointer + current_frame_range[0]
    timeline_select_frame.value = frame_n
    report_per_image_table.read_json(
        {
            "columns": report_per_image_table_columns,
            "data": [get_report_per_image_row(report_to_dict(current_report), frame_n)],
        }
    )
    show_images(frame_n)


def apply_timeline_filters(*args):
    global gt_video_info
    global current_frame_range
    filters = get_timeline_filters()
    report = report_to_dict(current_report)
    intervals, colors = get_intervals_with_colors(report, filters, current_frame_range)
    timeline.set(current_frame_range[1] - current_frame_range[0] + 1, intervals, colors)


for filter_metric_widget in timeline_filters_widgets.values():

    def cb_factory(filter_metric_widget):
        def metric_switched_cb(is_switched):
            if is_switched:
                filter_metric_widget["input"].enable()
            else:
                filter_metric_widget["input"].disable()
            apply_timeline_filters()

        return metric_switched_cb

    filter_metric_widget["input"].value_changed(apply_timeline_filters)
    filter_metric_widget["switch"].value_changed(cb_factory(filter_metric_widget))
    filter_metric_widget["input"].disable()
