#!/usr/bin/env python3
"""
Generate HTML Report from Markdown Executive Summary
"""

import markdown
import re
from datetime import datetime

def create_html_report():
    """Convert markdown executive summary to styled HTML report"""
    
    # Read the markdown file
    with open('Executive_summary/waze_churn_analysis_executive_summary.md', 'r', encoding='utf-8') as file:
        markdown_content = file.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'toc', 'codehilite'])
    html_content = md.convert(markdown_content)
    
    # Wrap images with professional containers
    html_content = re.sub(
        r'<p><img src="([^"]*)" alt="([^"]*)" /></p>',
        r'<div class="image-container"><img src="\1" alt="\2" /><p class="image-caption">\2</p></div>',
        html_content
    )
    
    # Create professional HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waze User Churn Analysis - Executive Summary</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        
        h3 {{
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        
        h4 {{
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 12px;
            font-size: 1.2em;
            font-weight: 600;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-bottom: 15px;
            padding-left: 30px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e8f4f8;
        }}
        
        .emoji-section {{
            display: inline-block;
            margin-right: 10px;
            font-size: 1.2em;
        }}
        
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .success {{
            background-color: #d4edda;
            padding: 15px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .info {{
            background-color: #d1ecf1;
            padding: 15px;
            border-left: 4px solid #17a2b8;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .warning {{
            background-color: #f8d7da;
            padding: 15px;
            border-left: 4px solid #dc3545;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
            color: #e83e8c;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            h2 {{
                font-size: 1.5em;
            }}
            
            table {{
                font-size: 0.9em;
            }}
            
            th, td {{
                padding: 8px 10px;
            }}
            
            /* Mobile-optimized image styling */
            img[alt*="Waze EDA"], img[alt*="Model Evaluation"] {{
                max-width: 95%;
                max-height: 500px;
                padding: 10px;
            }}
            
            .image-container {{
                padding: 10px;
                margin: 20px 0;
            }}
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #dee2e6;
        }}
        
        .toc h3 {{
            color: #495057;
            margin-bottom: 15px;
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin-bottom: 5px;
        }}
        
        .toc a {{
            color: #007bff;
            text-decoration: none;
            padding: 5px 10px;
            display: block;
            border-radius: 4px;
            transition: background-color 0.3s;
        }}
        
        .toc a:hover {{
            background-color: #e9ecef;
        }}
        
        /* Image styling for plots */
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            background: white;
            padding: 15px;
        }}
        
        /* Specific styling for analysis plots */
        img[alt*="Waze EDA"], img[alt*="Model Evaluation"] {{
            max-width: 90%;
            width: auto;
            max-height: 700px;
            object-fit: contain;
            border: 3px solid #3498db;
            border-radius: 12px;
            padding: 20px;
            background: #f8f9fa;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }}
        
        /* Container for images with captions */
        .image-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .image-caption {{
            font-style: italic;
            color: #666;
            margin-top: 15px;
            font-size: 0.95em;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Waze User Churn Analysis</h1>
            <div class="success">
                <strong>Executive Summary Report</strong><br>
                Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
            </div>
        </div>
        
        <div class="highlight">
            <h3>📊 Key Results at a Glance</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <div class="metric-card">
                    <div class="metric-value">17.7%</div>
                    <div class="metric-label">Overall Churn Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">75.4%</div>
                    <div class="metric-label">Best Model AUC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">14,299</div>
                    <div class="metric-label">Users Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">19</div>
                    <div class="metric-label">Features Engineered</div>
                </div>
            </div>
        </div>
        
        {html_content}
        
        <div class="footer">
            <p><strong>Waze User Churn Analysis Report</strong></p>
            <p>Generated using PACE methodology (Plan, Analyze, Construct, Execute)</p>
            <p>© 2024 Data Science Team - Confidential</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save the HTML report
    with open('Executive_summary/waze_churn_analysis_report.html', 'w', encoding='utf-8') as file:
        file.write(html_template)
    
    print("✅ HTML report generated successfully!")
    print("📄 File: waze_churn_analysis_report.html")
    print("🎯 Report includes:")
    print("   - Professional styling and responsive design")
    print("   - Key metrics dashboard")
    print("   - Complete analysis results")
    print("   - Business recommendations")
    print("   - Implementation roadmap")

if __name__ == "__main__":
    create_html_report() 