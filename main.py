import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import io


class ColorAnalyzer:
    """Deep color analysis for image blocks"""
    
    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get approximate color name from RGB"""
        r, g, b = rgb
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        # Determine color based on HSV
        if v < 0.2:
            return "Black"
        elif s < 0.1:
            if v > 0.9:
                return "White"
            elif v > 0.5:
                return "Light Gray"
            else:
                return "Dark Gray"
        
        hue_deg = h * 360
        if hue_deg < 15 or hue_deg >= 345:
            return "Red"
        elif hue_deg < 45:
            return "Orange"
        elif hue_deg < 75:
            return "Yellow"
        elif hue_deg < 165:
            return "Green"
        elif hue_deg < 255:
            return "Blue"
        elif hue_deg < 285:
            return "Purple"
        else:
            return "Pink"
    
    def analyze_block(self, block: np.ndarray) -> Dict:
        """Perform deep color analysis on a single block"""
        # Reshape image to be a list of pixels
        pixels = block.reshape(-1, 3)
        
        # Remove pure black/white if they're edge artifacts
        pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
        pixels = pixels[~np.all(pixels == [255, 255, 255], axis=1)]
        
        if len(pixels) == 0:
            pixels = block.reshape(-1, 3)
        
        # K-means clustering to find dominant colors
        n_colors = min(self.n_colors, len(pixels))
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get color percentages
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        colors_info = []
        for i, center in enumerate(kmeans.cluster_centers_):
            percentage = (label_counts[i] / total_pixels) * 100
            rgb = tuple(center.astype(int))
            colors_info.append({
                'rgb': rgb,
                'hex': self.rgb_to_hex(rgb),
                'name': self.get_color_name(rgb),
                'percentage': round(percentage, 2)
            })
        
        # Sort by percentage
        colors_info.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Calculate average brightness and saturation
        avg_hsv = colorsys.rgb_to_hsv(
            np.mean(pixels[:, 0])/255,
            np.mean(pixels[:, 1])/255,
            np.mean(pixels[:, 2])/255
        )
        
        return {
            'primary_color': colors_info[0],
            'color_palette': colors_info,
            'brightness': round(avg_hsv[2] * 100, 2),
            'saturation': round(avg_hsv[1] * 100, 2),
            'dominant_hue': round(avg_hsv[0] * 360, 2)
        }


class ImageGridAnalyzer:
    """Main class for grid-based image analysis"""
    
    def __init__(self, image_path: str, rows: int, cols: int, output_folder: str = "output"):
        self.image_path = image_path
        self.rows = rows
        self.cols = cols
        self.output_folder = output_folder
        self.color_analyzer = ColorAnalyzer()
        
        # Create output folder
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        self.height, self.width = self.image.shape[:2]
        self.block_height = self.height // self.rows
        self.block_width = self.width // self.cols
        
        self.analysis_results = []
    
    def divide_into_grid(self) -> List[Tuple[np.ndarray, int, int]]:
        """Divide image into grid blocks"""
        blocks = []
        for row in range(self.rows):
            for col in range(self.cols):
                y_start = row * self.block_height
                y_end = (row + 1) * self.block_height if row < self.rows - 1 else self.height
                x_start = col * self.block_width
                x_end = (col + 1) * self.block_width if col < self.cols - 1 else self.width
                
                block = self.image[y_start:y_end, x_start:x_end]
                blocks.append((block, row, col))
        
        return blocks
    
    def analyze_all_blocks(self):
        """Analyze all grid blocks"""
        blocks = self.divide_into_grid()
        total_blocks = len(blocks)
        
        print(f"Analyzing {total_blocks} blocks ({self.rows}x{self.cols} grid)...")
        
        for idx, (block, row, col) in enumerate(blocks):
            print(f"Analyzing block {idx + 1}/{total_blocks} (Row {row + 1}, Col {col + 1})...")
            analysis = self.color_analyzer.analyze_block(block)
            analysis['block_id'] = idx
            analysis['position'] = {'row': row, 'col': col}
            self.analysis_results.append(analysis)
    
    def create_grid_visualization(self, blocks_per_viz: int = 6) -> List[str]:
        """Create visualization images showing color analysis"""
        total_blocks = len(self.analysis_results)
        num_visualizations = (total_blocks + blocks_per_viz - 1) // blocks_per_viz
        viz_paths = []
        
        for viz_idx in range(num_visualizations):
            start_idx = viz_idx * blocks_per_viz
            end_idx = min(start_idx + blocks_per_viz, total_blocks)
            blocks_to_viz = self.analysis_results[start_idx:end_idx]
            
            # Calculate figure height based on number of blocks
            fig_height = max(12, len(blocks_to_viz) * 2.5)
            
            # Create figure with better spacing
            fig = plt.figure(figsize=(24, fig_height))
            gs = fig.add_gridspec(len(blocks_to_viz), 3, 
                                width_ratios=[1.2, 2, 2.5], 
                                hspace=0.6, 
                                wspace=0.4,
                                top=0.93,
                                bottom=0.05,
                                left=0.05,
                                right=0.95)
            
            for i, block_data in enumerate(blocks_to_viz):
                block_id = block_data['block_id']
                row, col = block_data['position']['row'], block_data['position']['col']
                
                # Get the actual block image
                y_start = row * self.block_height
                y_end = (row + 1) * self.block_height if row < self.rows - 1 else self.height
                x_start = col * self.block_width
                x_end = (col + 1) * self.block_width if col < self.cols - 1 else self.width
                block_img = self.image[y_start:y_end, x_start:x_end]
                
                # 1. Show block image
                ax1 = fig.add_subplot(gs[i, 0])
                ax1.imshow(block_img)
                ax1.set_title(f"Block {block_id + 1}\n(R{row + 1}, C{col + 1})", 
                            fontsize=11, fontweight='bold', pad=10)
                ax1.axis('off')
                
                # 2. Show color palette
                ax2 = fig.add_subplot(gs[i, 1])
                palette = block_data['color_palette']
                colors = [np.array(c['rgb'])/255 for c in palette]
                percentages = [c['percentage'] for c in palette]
                
                # Create color bars with labels
                y_positions = range(len(colors))
                bars = ax2.barh(y_positions, percentages, color=colors, height=0.7)
                
                # Add percentage labels on bars
                for j, (bar, pct) in enumerate(zip(bars, percentages)):
                    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                            f'{pct:.1f}%', va='center', fontsize=8, fontweight='bold')
                
                ax2.set_xlim(0, max(100, max(percentages) * 1.2))
                ax2.set_ylim(-0.5, len(colors) - 0.5)
                ax2.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
                ax2.set_title('Color Distribution', fontsize=11, fontweight='bold', pad=10)
                ax2.set_yticks(y_positions)
                ax2.set_yticklabels([f"{c['name']}" for c in palette], fontsize=9)
                ax2.grid(axis='x', alpha=0.3)
                ax2.invert_yaxis()
                
                # 3. Show analysis details in a more compact format
                ax3 = fig.add_subplot(gs[i, 2])
                ax3.axis('off')
                
                # Create compact text summary with better formatting
                primary = block_data['primary_color']
                
                # Split into sections for better readability
                header_text = f"PRIMARY COLOR: {primary['name']}\nHex: {primary['hex']} | Coverage: {primary['percentage']}%"
                
                palette_text = "COLOR PALETTE:\n" + "\n".join([
                    f"{idx}. {c['name'][:12]:12} {c['hex']:8} {c['percentage']:5.1f}%" 
                    for idx, c in enumerate(palette[:5], 1)
                ])
                
                metrics_text = f"ANALYSIS METRICS:\nBrightness: {block_data['brightness']:5.1f}%\nSaturation: {block_data['saturation']:5.1f}%\nDominant Hue: {block_data['dominant_hue']:5.1f}°"
                
                # Position text sections with proper spacing
                ax3.text(0.02, 0.95, header_text, transform=ax3.transAxes,
                        fontsize=10, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
                
                ax3.text(0.02, 0.70, palette_text, transform=ax3.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                
                ax3.text(0.02, 0.25, metrics_text, transform=ax3.transAxes,
                        fontsize=9, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
            
            # Add main title with better positioning
            fig.suptitle(f'Color Analysis - Visualization {viz_idx + 1}/{num_visualizations}\n'
                        f'Blocks {start_idx + 1}-{end_idx} of {total_blocks}',
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Save visualization with higher DPI for better quality
            viz_path = os.path.join(self.output_folder, f'visualization_{viz_idx + 1}.png')
            plt.savefig(viz_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            viz_paths.append(viz_path)
            print(f"Created visualization {viz_idx + 1}/{num_visualizations}: {viz_path}")
        
        return viz_paths
    
    def create_grid_overlay(self) -> str:
        """Create an image showing the grid overlay with primary colors"""
        overlay = self.image.copy()
        
        # Draw grid lines and fill with primary colors (semi-transparent)
        for block_data in self.analysis_results:
            row, col = block_data['position']['row'], block_data['position']['col']
            primary_color = block_data['primary_color']['rgb']
            
            y_start = row * self.block_height
            y_end = (row + 1) * self.block_height if row < self.rows - 1 else self.height
            x_start = col * self.block_width
            x_end = (col + 1) * self.block_width if col < self.cols - 1 else self.width
            
            # Create semi-transparent overlay
            overlay_block = overlay[y_start:y_end, x_start:x_end].copy()
            color_overlay = np.full_like(overlay_block, primary_color)
            overlay[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                overlay_block, 0.6, color_overlay, 0.4, 0
            )
        
        # Draw grid lines
        for i in range(1, self.rows):
            y = i * self.block_height
            cv2.line(overlay, (0, y), (self.width, y), (255, 255, 255), 2)
        
        for i in range(1, self.cols):
            x = i * self.block_width
            cv2.line(overlay, (x, 0), (x, self.height), (255, 255, 255), 2)
        
        # Save overlay
        overlay_path = os.path.join(self.output_folder, 'grid_overlay.png')
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_path, overlay_bgr)
        print(f"Created grid overlay: {overlay_path}")
        
        return overlay_path
    
    def save_analysis_json(self) -> str:
        """Save analysis results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for block in self.analysis_results:
            json_block = {
                'block_id': int(block['block_id']),
                'position': block['position'],
                'primary_color': {
                    'rgb': [int(x) for x in block['primary_color']['rgb']],
                    'hex': block['primary_color']['hex'],
                    'name': block['primary_color']['name'],
                    'percentage': float(block['primary_color']['percentage'])
                },
                'color_palette': [
                    {
                        'rgb': [int(x) for x in c['rgb']],
                        'hex': c['hex'],
                        'name': c['name'],
                        'percentage': float(c['percentage'])
                    } for c in block['color_palette']
                ],
                'brightness': float(block['brightness']),
                'saturation': float(block['saturation']),
                'dominant_hue': float(block['dominant_hue'])
            }
            json_results.append(json_block)
        
        analysis_data = {
            'metadata': {
                'image_path': self.image_path,
                'grid_size': {'rows': self.rows, 'cols': self.cols},
                'total_blocks': len(self.analysis_results),
                'image_dimensions': {'width': self.width, 'height': self.height},
                'analysis_date': datetime.now().isoformat()
            },
            'blocks': json_results
        }
        
        json_path = os.path.join(self.output_folder, 'analysis_data.json')
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"Saved analysis data: {json_path}")
        return json_path
    
    def create_watermark_design(self, width: int, height: int, color_rgb: Tuple[int, int, int], opacity: float = 0.3) -> Image.Image:
        """Create a watermark design with the specified color"""
        # Create transparent image
        watermark = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Convert RGB to RGBA with opacity
        r, g, b = color_rgb
        alpha = int(255 * opacity)
        color_rgba = (r, g, b, alpha)
        
        # Create a geometric watermark design
        # Design: Overlapping circles and lines pattern
        
        # Calculate sizes based on block dimensions
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 3
        
        # Draw concentric circles
        for i in range(3):
            radius = max_radius - (i * max_radius // 4)
            if radius > 0:
                draw.ellipse(
                    [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                    outline=color_rgba,
                    width=max(2, width // 100)
                )
        
        # Draw diagonal lines
        line_width = max(2, width // 150)
        num_lines = 8
        for i in range(num_lines):
            angle = (360 / num_lines) * i
            # Calculate line endpoints
            length = min(width, height) // 2.5
            x1 = center_x + int(length * np.cos(np.radians(angle)))
            y1 = center_y + int(length * np.sin(np.radians(angle)))
            x2 = center_x - int(length * np.cos(np.radians(angle)))
            y2 = center_y - int(length * np.sin(np.radians(angle)))
            draw.line([x1, y1, x2, y2], fill=color_rgba, width=line_width)
        
        # Draw a central star/diamond shape
        star_size = max_radius // 2
        points = []
        for i in range(8):
            angle = (360 / 8) * i
            if i % 2 == 0:
                radius = star_size
            else:
                radius = star_size // 2
            x = center_x + int(radius * np.cos(np.radians(angle - 90)))
            y = center_y + int(radius * np.sin(np.radians(angle - 90)))
            points.append((x, y))
        
        draw.polygon(points, outline=color_rgba, width=max(2, width // 100))
        
        return watermark
    
    def apply_watermarks_to_image(self, opacity: float = 0.3) -> str:
        """Apply watermarks to each block of the image using block's primary color"""
        print("\nApplying watermarks to blocks...")
        
        # Convert numpy array to PIL Image
        watermarked_image = Image.fromarray(self.image.astype('uint8'), 'RGB')
        
        for block_data in self.analysis_results:
            block_id = block_data['block_id']
            row, col = block_data['position']['row'], block_data['position']['col']
            primary_color_rgb = block_data['primary_color']['rgb']
            
            # Calculate block boundaries
            y_start = row * self.block_height
            y_end = (row + 1) * self.block_height if row < self.rows - 1 else self.height
            x_start = col * self.block_width
            x_end = (col + 1) * self.block_width if col < self.cols - 1 else self.width
            
            block_width = x_end - x_start
            block_height = y_end - y_start
            
            # Create watermark for this block
            watermark = self.create_watermark_design(block_width, block_height, primary_color_rgb, opacity)
            
            # Paste watermark onto the image
            watermarked_image.paste(watermark, (x_start, y_start), watermark)
            
            print(f"Applied watermark to Block {block_id + 1} with color {block_data['primary_color']['hex']}")
        
        # Save watermarked image
        watermarked_path = os.path.join(self.output_folder, 'watermarked_image.png')
        watermarked_image.save(watermarked_path, 'PNG')
        print(f"\nCreated watermarked image: {watermarked_path}")
        
        return watermarked_path
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("IMAGE GRID COLOR ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Image: {self.image_path}")
        report_lines.append(f"Grid Size: {self.rows} rows x {self.cols} columns")
        report_lines.append(f"Total Blocks: {len(self.analysis_results)}")
        report_lines.append(f"Image Dimensions: {self.width} x {self.height} pixels")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for block in self.analysis_results:
            block_id = block['block_id']
            row, col = block['position']['row'], block['position']['col']
            primary = block['primary_color']
            
            report_lines.append(f"BLOCK {block_id + 1} (Row {row + 1}, Column {col + 1})")
            report_lines.append("-" * 80)
            report_lines.append(f"Primary Color: {primary['name']} ({primary['hex']}) - {primary['percentage']}%")
            report_lines.append(f"Brightness: {block['brightness']}% | Saturation: {block['saturation']}% | Hue: {block['dominant_hue']}°")
            report_lines.append("\nColor Palette:")
            for i, color in enumerate(block['color_palette'][:5], 1):
                report_lines.append(f"  {i}. {color['name']:15} {color['hex']:8} - {color['percentage']:6.2f}%")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report_path = os.path.join(self.output_folder, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Generated summary report: {report_path}")
        return report_path
    
    def run_full_analysis(self, create_visualizations: bool = True, apply_watermarks: bool = True, watermark_opacity: float = 0.3):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING IMAGE GRID COLOR ANALYSIS")
        print("="*80 + "\n")
        
        # Step 1: Analyze blocks
        self.analyze_all_blocks()
        
        # Step 2: Create visualizations (optional)
        viz_paths = []
        if create_visualizations:
            print("\nCreating visualizations...")
            viz_paths = self.create_grid_visualization()
        
        # Step 3: Create grid overlay
        print("\nCreating grid overlay...")
        overlay_path = self.create_grid_overlay()
        
        # Step 4: Apply watermarks (optional)
        watermarked_path = None
        if apply_watermarks:
            print("\nApplying watermarks...")
            watermarked_path = self.apply_watermarks_to_image(opacity=watermark_opacity)
        
        # Step 5: Save JSON data
        print("\nSaving analysis data...")
        json_path = self.save_analysis_json()
        
        # Step 6: Generate report
        print("\nGenerating summary report...")
        report_path = self.generate_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nOutput folder: {self.output_folder}")
        print(f"\nGenerated files:")
        if viz_paths:
            print(f"  - {len(viz_paths)} visualization image(s)")
        print(f"  - 1 grid overlay image")
        if watermarked_path:
            print(f"  - 1 watermarked image")
        print(f"  - 1 JSON data file")
        print(f"  - 1 text report")
        print("\n" + "="*80 + "\n")


def main():
    """Main function to run the script"""
    print("\n" + "="*80)
    print("IMAGE GRID COLOR ANALYZER")
    print("="*80 + "\n")
    
    # Get user inputs
    image_path = input("Enter the path to your image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        rows = int(input("Enter number of rows for the grid: ").strip())
        cols = int(input("Enter number of columns for the grid: ").strip())
        
        if rows <= 0 or cols <= 0:
            print("Error: Rows and columns must be positive integers")
            return
    except ValueError:
        print("Error: Please enter valid integers for rows and columns")
        return
    
    output_folder = input("Enter output folder name (default: 'output'): ").strip() or "output"
    
    # Ask about visualizations
    viz_choice = input("Create detailed visualizations? (yes/no, default: yes): ").strip().lower()
    create_visualizations = viz_choice != 'no'
    
    # Ask about watermarking
    watermark_choice = input("Apply watermarks to blocks? (yes/no, default: yes): ").strip().lower()
    apply_watermarks = watermark_choice != 'no'
    
    watermark_opacity = 0.3
    if apply_watermarks:
        opacity_input = input("Enter watermark opacity (0.1-1.0, default: 0.3): ").strip()
        if opacity_input:
            try:
                watermark_opacity = float(opacity_input)
                watermark_opacity = max(0.1, min(1.0, watermark_opacity))  # Clamp between 0.1 and 1.0
            except ValueError:
                print("Invalid opacity value, using default 0.3")
                watermark_opacity = 0.3
    
    # Create analyzer and run
    try:
        analyzer = ImageGridAnalyzer(image_path, rows, cols, output_folder)
        analyzer.run_full_analysis(create_visualizations=create_visualizations, apply_watermarks=apply_watermarks, watermark_opacity=watermark_opacity)
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
