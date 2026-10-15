# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Show one entry per release and keep the shared guide out of the sidebar."""

from pathlib import PurePosixPath

from mkdocs.structure.nav import Navigation
from mkdocs.structure.pages import Page


def on_nav(nav: Navigation, **kwargs) -> Navigation:
    def collapse(items):
        for position, item in enumerate(items):
            if not item.is_section:
                continue

            pages = item.children
            if not all(isinstance(child, Page) for child in pages):
                collapse(pages)
                continue

            index = next((child for child in pages if child.is_index), None)
            if index is None:
                continue
            directory = PurePosixPath(index.file.src_uri).parent
            if directory.parent != PurePosixPath("release"):
                continue
            if any(
                PurePosixPath(child.file.src_uri).parent != directory for child in pages
            ):
                continue
            names = {PurePosixPath(child.file.src_uri).name for child in pages}
            if names not in (
                {"README.md", "en.md", "zh.md"},
                {"README.md", "common_en.md", "common_zh.md"},
            ):
                continue

            for child in pages:
                child.title = item.title
                child.parent = item.parent

            # Material's nav-item macro selects the current page from `pages`.
            # All variants stay in nav.pages, preserving their active ancestors
            # and the normal page TOC, including on mobile.
            index.pages = pages  # type: ignore[attr-defined]
            items[position] = index

        # Keep common pages in nav.pages with their active ancestors, so links
        # and language switches retain the sidebar without a separate guide row.
        items[:] = [
            item
            for item in items
            if not (
                isinstance(item, Page)
                and PurePosixPath(item.file.src_uri).parent
                == PurePosixPath("release/common")
            )
        ]

    collapse(nav.items)
    return nav
