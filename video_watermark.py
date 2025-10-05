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
from datetime import datetime
import io
import time
import warnings

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*Number of distinct clusters.*')


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
        
        # Optimize: Sample pixels if block is too large (>10000 pixels)
        if len(pixels) > 10000:
            # Randomly sample pixels to speed up k-means
            sample_size = 10000
            indices = np.random.choice(len(pixels), size=sample_size, replace=False)
            pixels = pixels[indices]
        
        # Determine number of unique colors in the sample
        unique_colors = len(np.unique(pixels, axis=0))
        n_colors = min(self.n_colors, len(pixels), unique_colors)
        
        # If only 1 unique color, handle it directly without k-means
        if n_colors <= 1:
            avg_color = np.mean(pixels, axis=0).astype(int)
            rgb = tuple(avg_color)
            return {
                'primary_color': {
                    'rgb': rgb,
                    'hex': self.rgb_to_hex(rgb),
                    'name': self.get_color_name(rgb),
                    'percentage': 100.0
                },
                'color_palette': [{
                    'rgb': rgb,
                    'hex': self.rgb_to_hex(rgb),
                    'name': self.get_color_name(rgb),
                    'percentage': 100.0
                }],
                'brightness': round(colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)[2] * 100, 2),
                'saturation': round(colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)[1] * 100, 2),
                'dominant_hue': round(colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)[0] * 360, 2)
            }
        
        # K-means clustering to find dominant colors
        # Suppress warnings with proper parameters
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            kmeans = KMeans(
                n_clusters=n_colors, 
                random_state=42, 
                n_init='auto',  # Use 'auto' to suppress FutureWarning
                max_iter=300,   # Maximum iterations for convergence
                tol=1e-4,       # Tolerance for convergence
                algorithm='lloyd'  # Explicit algorithm selection
            )
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


class VideoWatermarkProcessor:
    """Process video frames with grid-based watermarking"""
    
    def __init__(self, video_path: str, rows: int, cols: int, output_folder: str = "video_output"):
        self.video_path = video_path
        self.rows = rows
        self.cols = cols
        self.output_folder = output_folder
        self.color_analyzer = ColorAnalyzer()
        
        # Cache for watermark designs to avoid recreating identical watermarks
        self.watermark_cache = {}
        
        # Create output folder
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Open video
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Could not open video from {video_path}")
        
        # Get video properties
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.block_height = self.height // self.rows
        self.block_width = self.width // self.cols
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.fps:.2f} seconds")
        print(f"  Block Size: {self.block_width}x{self.block_height}")
    
    def clear_cache(self):
        """Clear the watermark cache to free memory"""
        self.watermark_cache.clear()
    
    def analyze_frame_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Analyze all blocks in a single frame"""
        analysis_results = []
        
        for row in range(self.rows):
            for col in range(self.cols):
                y_start = row * self.block_height
                y_end = (row + 1) * self.block_height if row < self.rows - 1 else self.height
                x_start = col * self.block_width
                x_end = (col + 1) * self.block_width if col < self.cols - 1 else self.width
                
                block = frame[y_start:y_end, x_start:x_end]
                analysis = self.color_analyzer.analyze_block(block)
                analysis['position'] = {'row': row, 'col': col}
                analysis_results.append(analysis)
        
        return analysis_results
    
    def create_watermark_design(self, width: int, height: int, color_rgb: Tuple[int, int, int], opacity: float = 0.3) -> Image.Image:
        """Create a watermark design with the specified color (with caching)"""
        # Create cache key from parameters
        cache_key = (width, height, color_rgb, opacity)
        
        # Return cached watermark if it exists
        if cache_key in self.watermark_cache:
            return self.watermark_cache[cache_key].copy()
        
        # Create transparent image
        watermark = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Convert RGB to RGBA with opacity
        r, g, b = color_rgb
        alpha = int(255 * opacity)
        color_rgba = (r, g, b, alpha)
        
        # Create a geometric watermark design
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
        
        # Cache the watermark for reuse
        self.watermark_cache[cache_key] = watermark.copy()
        
        return watermark
    
    def apply_watermark_to_frame(self, frame: np.ndarray, analysis_results: List[Dict], opacity: float = 0.3) -> np.ndarray:
        """Apply watermarks to a single frame (optimized)"""
        # Convert numpy array to PIL Image (reuse conversion)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        watermarked_frame = Image.fromarray(frame_rgb.astype('uint8'))
        
        for block_data in analysis_results:
            row, col = block_data['position']['row'], block_data['position']['col']
            primary_color_rgb = block_data['primary_color']['rgb']
            
            # Calculate block boundaries
            y_start = row * self.block_height
            y_end = (row + 1) * self.block_height if row < self.rows - 1 else self.height
            x_start = col * self.block_width
            x_end = (col + 1) * self.block_width if col < self.cols - 1 else self.width
            
            block_width = x_end - x_start
            block_height = y_end - y_start
            
            # Create watermark for this block (uses cache internally)
            watermark = self.create_watermark_design(block_width, block_height, primary_color_rgb, opacity)
            
            # Paste watermark onto the frame
            watermarked_frame.paste(watermark, (x_start, y_start), watermark)
        
        # Convert back to BGR for OpenCV
        watermarked_array = np.array(watermarked_frame)
        watermarked_bgr = cv2.cvtColor(watermarked_array, cv2.COLOR_RGB2BGR)
        
        return watermarked_bgr
    
    def process_video(self, opacity: float = 0.3, analyze_every_n_frames: int = 1):
        """Process entire video with watermarking"""
        print("\n" + "="*80)
        print("STARTING VIDEO WATERMARKING")
        print("="*80 + "\n")
        
        # Prepare output video path
        video_name = Path(self.video_path).stem
        output_path = os.path.join(self.output_folder, f'{video_name}_watermarked.mp4')
        
        # Create video writer
        out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        processed_count = 0
        last_analysis = None
        start_time = time.time()
        
        print(f"Processing video: {self.video_path}")
        print(f"Output: {output_path}")
        print(f"Analyzing every {analyze_every_n_frames} frame(s)\n")
        
        def print_progress_bar(current, total, start_time, analyzed):
            """Print a visual progress bar with stats"""
            progress = current / total
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Calculate time stats
            elapsed = time.time() - start_time
            if current > 0:
                fps = current / elapsed
                eta_seconds = (total - current) / fps if fps > 0 else 0
                eta_mins = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
            else:
                fps = 0
                eta_mins = 0
                eta_secs = 0
            
            # Print progress bar
            print(f"\r[{bar}] {progress*100:.1f}% | {current}/{total} frames | "
                  f"Speed: {fps:.1f} fps | ETA: {eta_mins:02d}:{eta_secs:02d} | "
                  f"Analyzed: {analyzed}", end='', flush=True)
        
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze frame (or reuse last analysis for performance)
            if (frame_count - 1) % analyze_every_n_frames == 0 or last_analysis is None:
                # Convert BGR to RGB for analysis
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_analysis = self.analyze_frame_blocks(frame_rgb)
                processed_count += 1
            
            # Apply watermark using current analysis
            watermarked_frame = self.apply_watermark_to_frame(frame, last_analysis, opacity)
            
            # Write frame
            out.write(watermarked_frame)
            
            # Progress update (every frame for smooth progress bar)
            print_progress_bar(frame_count, self.total_frames, start_time, processed_count)
        
        # Print final newline after progress bar
        print()
        
        # Cleanup
        self.video.release()
        out.release()
        
        print("\n" + "="*80)
        print("VIDEO WATERMARKING COMPLETE!")
        print("="*80)
        print(f"\nOutput video: {output_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames analyzed: {processed_count}")
        print(f"Grid size: {self.rows}x{self.cols}")
        print(f"Watermark opacity: {opacity}")
        print(f"Watermark cache size: {len(self.watermark_cache)} unique designs")
        print(f"Cache efficiency: {(frame_count * self.rows * self.cols - len(self.watermark_cache)) / max(1, frame_count * self.rows * self.cols) * 100:.1f}% reuse")
        print("\n" + "="*80 + "\n")
        
        return output_path
    
    def save_sample_frame_analysis(self, frame_number: int = 0, opacity: float = 0.3):
        """Save a sample frame with analysis for verification"""
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        
        if not ret:
            print(f"Could not read frame {frame_number}")
            return
        
        # Analyze frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis_results = self.analyze_frame_blocks(frame_rgb)
        
        # Apply watermark
        watermarked_frame = self.apply_watermark_to_frame(frame, analysis_results, opacity)
        
        # Save original and watermarked frames
        original_path = os.path.join(self.output_folder, f'sample_frame_{frame_number}_original.png')
        watermarked_path = os.path.join(self.output_folder, f'sample_frame_{frame_number}_watermarked.png')
        
        cv2.imwrite(original_path, frame)
        cv2.imwrite(watermarked_path, watermarked_frame)
        
        print(f"\nSaved sample frames:")
        print(f"  Original: {original_path}")
        print(f"  Watermarked: {watermarked_path}")
        
        # Save analysis data
        analysis_data = {
            'frame_number': frame_number,
            'grid_size': {'rows': self.rows, 'cols': self.cols},
            'total_blocks': len(analysis_results),
            'opacity': opacity,
            'blocks': []
        }
        
        for idx, block in enumerate(analysis_results):
            analysis_data['blocks'].append({
                'block_id': idx,
                'position': block['position'],
                'primary_color': {
                    'rgb': [int(x) for x in block['primary_color']['rgb']],
                    'hex': block['primary_color']['hex'],
                    'name': block['primary_color']['name']
                }
            })
        
        json_path = os.path.join(self.output_folder, f'sample_frame_{frame_number}_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"  Analysis: {json_path}")


def main():
    """Main function to run the video watermarking script"""
    print("\n" + "="*80)
    print("VIDEO WATERMARK PROCESSOR")
    print("="*80 + "\n")
    
    # Get user inputs
    video_path = input("Enter the path to your video: ").strip()
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
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
    
    output_folder = input("Enter output folder name (default: 'video_output'): ").strip() or "video_output"
    
    # Ask about watermark opacity
    opacity_input = input("Enter watermark opacity (0.1-1.0, default: 0.12): ").strip()
    watermark_opacity = 0.12
    if opacity_input:
        try:
            watermark_opacity = float(opacity_input)
            watermark_opacity = max(0.1, min(1.0, watermark_opacity))
        except ValueError:
            print("Invalid opacity value, using default 0.12")
            watermark_opacity = 0.12
    
    # Ask about analysis frequency
    analyze_freq_input = input("Analyze every N frames (1=every frame, 5=every 5th frame, default: 1): ").strip()
    analyze_every_n_frames = 1
    if analyze_freq_input:
        try:
            analyze_every_n_frames = int(analyze_freq_input)
            analyze_every_n_frames = max(1, analyze_every_n_frames)
        except ValueError:
            print("Invalid value, using default 1")
            analyze_every_n_frames = 1
    
    # Ask about sample frame
    save_sample = input("Save sample frame analysis? (yes/no, default: yes): ").strip().lower()
    save_sample_frame = save_sample != 'no'
    
    # Create processor and run
    try:
        processor = VideoWatermarkProcessor(video_path, rows, cols, output_folder)
        
        # Save sample frame if requested
        if save_sample_frame:
            print("\nGenerating sample frame analysis...")
            processor.save_sample_frame_analysis(frame_number=0, opacity=watermark_opacity)
        
        # Process video
        output_path = processor.process_video(opacity=watermark_opacity, analyze_every_n_frames=analyze_every_n_frames)
        
        print(f"\n✓ Video watermarking completed successfully!")
        print(f"✓ Output saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
