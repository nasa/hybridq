#!/bin/sh

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

# Enter docs folder
echo "# Enter docs folder." >&2
cd docs/

# Get pdoc
PDOC=${PDOC:-pdoc3}

# Generate HTML
echo "# Generate HTML." >&2
$PDOC --skip-errors -f --html hybridq 2>/dev/null

# Generate PDF Markdown
echo "# Generate PDF Markdown." >&2
$PDOC --skip-errors --pdf hybridq > html/hybridq/pdf.md 2>/dev/null

# Enter HTML folder
echo "# Enter HTML folder." >&2
cd html/hybridq

# Define pandoc
PANDOC=${PANDOC:-pandoc}

# Generate PDF
echo "# Generate PDF." >&2
$PANDOC --metadata=title:"HybridQ Documentation" \
        --from=markdown+abbreviations+tex_math_single_backslash \
        --pdf-engine=xelatex --variable=mainfont:"DejaVu Sans" \
        --toc --toc-depth=4 --output=../../hybridq.pdf pdf.md 2>/dev/null

# Remove PDF Markdown
echo "# Remove PDF Markdown." >&2
rm pdf.md
