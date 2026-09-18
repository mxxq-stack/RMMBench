#!/usr/bin/env python3
"""
Add top_site and bottom_site to a MuJoCo XML model based on its OBJ mesh bounding box.

Usage:
    python add_top_bottom_sites.py <directory_path>

Example:
    python add_top_bottom_sites.py /Users/lh/work/vlabench_etc/VLABench/VLABench/assets/obj/meshes/tools/tools_holder/tool_holder_0

The script expects the directory to contain:
    - <name>.xml (MuJoCo XML file)
    - <name>.obj (mesh file referenced in XML)

It will:
    1. Parse the OBJ to find the bounding box
    2. Read the XML to find the mesh scale
    3. Calculate top/bottom center points in body local coordinates
    4. Insert top_site and bottom_site before </body>
"""
import os
import sys
import glob
import xml.etree.ElementTree as ET

PATH = "/Users/lh/work/vlabench_etc/VLABench/VLABench/assets/obj/meshes/small_table/small_table_0"

def parse_obj_vertices(obj_path):
    """Parse OBJ file and return list of (x, y, z) vertices."""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
    return vertices


def get_mesh_scale_from_xml(xml_path):
    """Extract mesh scale from XML. Returns (scale_x, scale_y, scale_z) or (1, 1, 1)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find mesh element in asset
    for mesh in root.iter('mesh'):
        scale_str = mesh.get('scale', '1 1 1')
        parts = scale_str.split()
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    
    return (1.0, 1.0, 1.0)


def calculate_bounding_box(vertices, scale):
    """Calculate bounding box center and dimensions after scaling.
    
    Returns top/bottom positions as half of actual height (centered at origin).
    """
    if not vertices:
        return None
    
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)
    
    # Actual height after scaling (using Y axis for vertical direction)
    actual_height = (max_y - min_y) * scale[1]
    half_height = actual_height / 2
    
    # pos_x = 0, pos_y = +/- half_height, pos_z = 0
    # But in the site pos attribute, we put Y value in the Z position
    # because the body has euler rotation that makes Y the vertical axis
    return {
        'bottom': (0.0, 0.0, -half_height),
        'top': (0.0, 0.0, half_height),
        'dimensions': (
            (max_x - min_x) * scale[0],
            actual_height,
            (max_z - min_z) * scale[2]
        )
    }


def add_sites_to_xml(xml_path, bbox_info, site_size=0.005):
    """Add top_site and bottom_site to XML file.
    
    Sites are inserted after </body> and before </worldbody>.
    """
    with open(xml_path, 'r') as f:
        content = f.read()
    
    # Check if sites already exist
    if 'top_site' in content and 'bottom_site' in content:
        print(f"  Sites already exist in {os.path.basename(xml_path)}, skipping")
        return False
    
    # Find </body> followed by </worldbody> or just </body>
    # Insert sites AFTER </body> (between </body> and </worldbody>)
    body_end = content.find('    </body>')
    if body_end == -1:
        body_end = content.find('</body>')
        if body_end == -1:
            print(f"  ERROR: Could not find </body> tag in {os.path.basename(xml_path)}")
            return False
        # After </body>
        insert_pos = body_end + len('</body>')
        indent = '    '
    else:
        # After '    </body>'
        insert_pos = body_end + len('    </body>')
        indent = ''
    
    bottom = bbox_info['bottom']
    top = bbox_info['top']
    
    sites_xml = '\n' + indent + '    <site rgba="0 0 0 0" size="' + str(site_size) + '" pos="' + f'{bottom[0]:.15f} {bottom[1]:.15f} {bottom[2]:.15f}' + '" name="bottom_site" />\n' + indent + '    <site rgba="0 0 0 0" size="' + str(site_size) + '" pos="' + f'{top[0]:.15f} {top[1]:.15f} {top[2]:.15f}' + '" name="top_site" />\n' + indent + '  '
    
    new_content = content[:insert_pos] + sites_xml + content[insert_pos:]
    
    with open(xml_path, 'w') as f:
        f.write(new_content)
    
    return True


def process_directory(dir_path):
    """Process a directory containing XML and OBJ files."""
    dir_path = os.path.abspath(dir_path)
    
    if not os.path.isdir(dir_path):
        print(f"ERROR: {dir_path} is not a directory")
        return False
    
    # Find XML files (expecting one main XML file)
    xml_files = glob.glob(os.path.join(dir_path, '*.xml'))
    if not xml_files:
        print(f"ERROR: No XML files found in {dir_path}")
        return False
    
    # Use the first XML file that matches the directory name, or just the first one
    dir_name = os.path.basename(dir_path)
    xml_path = None
    for xf in xml_files:
        if os.path.basename(xf).replace('.xml', '') == dir_name:
            xml_path = xf
            break
    if not xml_path:
        xml_path = xml_files[0]
    
    xml_name = os.path.basename(xml_path).replace('.xml', '')
    obj_path = os.path.join(dir_path, f'{xml_name}.obj')
    
    if not os.path.exists(obj_path):
        # Try to find any OBJ file
        obj_files = glob.glob(os.path.join(dir_path, '*.obj'))
        if not obj_files:
            print(f"ERROR: No OBJ file found in {dir_path}")
            return False
        # Use the main OBJ (not collision ones)
        obj_files = [f for f in obj_files if '_collision_' not in os.path.basename(f)]
        if obj_files:
            obj_path = obj_files[0]
        else:
            obj_path = obj_files[0] if obj_files else glob.glob(os.path.join(dir_path, '*.obj'))[0]
    
    print(f"Processing: {dir_path}")
    print(f"  XML: {os.path.basename(xml_path)}")
    print(f"  OBJ: {os.path.basename(obj_path)}")
    
    # Parse OBJ
    vertices = parse_obj_vertices(obj_path)
    print(f"  Vertices: {len(vertices)}")
    
    if not vertices:
        print(f"  ERROR: No vertices found in OBJ")
        return False
    
    # Get scale from XML
    scale = get_mesh_scale_from_xml(xml_path)
    print(f"  Mesh scale: {scale}")
    
    # Calculate bounding box
    bbox = calculate_bounding_box(vertices, scale)
    if not bbox:
        print(f"  ERROR: Failed to calculate bounding box")
        return False
    
    print(f"  Bounding box dimensions: {bbox['dimensions']}")
    print(f"  Bottom: {bbox['bottom']}")
    print(f"  Top: {bbox['top']}")
    
    # Add sites
    success = add_sites_to_xml(xml_path, bbox)
    if success:
        # Verify
        try:
            ET.parse(xml_path)
            print(f"  Sites added successfully! XML valid.")
            return True
        except Exception as e:
            print(f"  ERROR: XML validation failed: {e}")
            return False
    else:
        print(f"  No changes needed or already has sites")
        return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nUsage: python {sys.argv[0]} <directory_path>")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    success = process_directory(dir_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
