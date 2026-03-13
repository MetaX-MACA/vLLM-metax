# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob

requires_files = glob.glob("requirements/*.txt")
requires_files += ["pyproject.toml"]
for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    if (
        "+metax" in "".join(lines).lower()
        or "+maca" in "".join(lines).lower()
        or "mcpy" in "".join(lines).lower()
    ):
        print("removed:")
        with open(file, "w") as f:
            for line in lines:
                if (
                    "+metax" not in line.lower()
                    and "+maca" not in line.lower()
                    and "mcpy" not in line.lower()
                ):
                    f.write(line)
                else:
                    print(line.strip())
    print(f"<<< done cleaning {file}")
    print()
