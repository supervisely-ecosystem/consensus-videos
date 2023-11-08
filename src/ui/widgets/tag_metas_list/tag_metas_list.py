from typing import List, Union
from supervisely.app.widgets import Widget, Button, generate_id
from supervisely import TagMeta, TagMetaCollection
from supervisely.app.content import DataJson, StateJson


class TagMetasList(Widget):
    def __init__(
        self,
        tag_metas: Union[List[TagMeta], TagMetaCollection] = [],
        multiple: bool = False,
        show_type_text: bool = True,
        limit_long_names: bool = False,
        max_height: str = None,
        widget_id: str = None,
    ):
        self._tag_metas = tag_metas
        self._multiple = multiple
        self._show_type_text = show_type_text
        self._limit_long_names = limit_long_names
        self._max_height = max_height

        if self._multiple:
            self._select_all_btn = Button(
                "Select all",
                button_type="text",
                show_loading=False,
                icon="zmdi zmdi-check-all",
                widget_id=generate_id(),
            )
            self._deselect_all_btn = Button(
                "Deselect all",
                button_type="text",
                show_loading=False,
                icon="zmdi zmdi-square-o",
                widget_id=generate_id(),
            )

            @self._select_all_btn.click
            def _select_all_btn_clicked():
                self.select_all()

            @self._deselect_all_btn.click
            def _deselect_all_btn_clicked():
                self.deselect_all()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "tag_metas": [tm.to_json() for tm in self._tag_metas],
        }

    def get_json_state(self):
        return {"selected": [False for _ in self._tag_metas]}

    def set(self, tag_metas: Union[List[TagMeta], TagMetaCollection]):
        self._tag_metas = tag_metas
        self.update_data()
        DataJson().send_changes()
        self.select_all()

    def select_all(self):
        StateJson()[self.widget_id]["selected"] = [True for _ in self._tag_metas]
        StateJson().send_changes()

    def deselect_all(self):
        StateJson()[self.widget_id]["selected"] = [False for _ in self._tag_metas]
        StateJson().send_changes()

    def get_all_tag_metas(self):
        return self._tag_metas

    def get_selected_tag_metas(self):
        selected = StateJson()[self.widget_id]["selected"]
        return [tm for i, tm in enumerate(self._tag_metas) if selected[i]]

    def select(self, tag_meta_name):
        try:
            idx = [tm for tm in self._tag_metas].index(tag_meta_name)
            StateJson()[self.widget_id]["selected"][idx] = True
            StateJson().send_changes()
        except:
            pass
