#!/bin/bash

# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
# 
# Copyright © 2021, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
# 
# The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. 
# 
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Define pdoc
PDOC=${PDOC:-pdoc3}

# Define pandoc
PANDOC=${PANDOC:-pandoc}

# Get random name
BACKUP=__backup.${RANDOM}${RANDOM}__

# Create folder
mkdir -p docs/$BACKUP >&2

# Backup docs folder
echo "# Backup docs folder." >&2
\mv -v docs/* docs/$BACKUP >&2

# Move back static folders
\mv -v docs/$BACKUP/nasa-cla docs/ >&2
\mv -v docs/$BACKUP/images docs/ >&2

# Generate HTML
echo "# Generate HTML." >&2
$PDOC --skip-errors -f --html -o docs/ hybridq 2>/dev/null
\mv -v docs/hybridq/* docs/

# Generate PDF Markdown
echo "# Generate PDF Markdown." >&2
$PDOC --skip-errors --pdf hybridq > docs/pdf.md 2>/dev/null

# Generate PDF
echo "# Generate PDF." >&2
$PANDOC --metadata=title:"HybridQ Documentation" \
        --from=markdown+abbreviations+tex_math_single_backslash \
        --pdf-engine=xelatex --variable=mainfont:"DejaVu Sans" \
        --toc --toc-depth=4 --output=docs/hybridq.pdf docs/pdf.md 2>/dev/null

# Remove PDF Markdown
echo "# Remove PDF Markdown." >&2
\rm -v docs/pdf.md >&2

# Remove Backup
echo "# Remove Backup." >&2
\rm -frv docs/$BACKUP
