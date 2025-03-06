import os
import subprocess
from datetime import datetime
from github_pages_setup import read_agent_data  # Import function to read agent data

# GitHub repository details
GITHUB_REPO = "zainkhatri/mcc-genai-guild"
BRANCH = "gh-pages"


def update_github_pages():
    """Updates the GitHub Pages site by replacing the leaderboard with new evaluation results."""
    try:
        os.chdir(os.path.join(os.path.dirname(__file__), "github-pages"))
    except FileNotFoundError:
        print("Error: 'github-pages' directory not found.")
        return
    except Exception as e:
        print(f"Error changing directory: {e}")
        return

    try:
        # Find the latest JSON file in the mizan_results directory
        mizan_results_dir = os.path.join(os.path.dirname(__file__), "Agent3/mizan_results")
        json_files = [f for f in os.listdir(mizan_results_dir) if f.endswith(".json")]
        if not json_files:
            print("Error: No JSON files found in the 'mizan_results' directory.")
            return
        latest_json_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(mizan_results_dir, f)))
        agent3_data = read_agent_data(os.path.join(mizan_results_dir, latest_json_file))
    except FileNotFoundError:
        print("Error: 'mizan_results' directory not found.")
        return
    except Exception as e:
        print(f"Error reading JSON files: {e}")
        return

    try:
        # Create a new leaderboard HTML table
        new_leaderboard_html = "<table border='1'><tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>Ethical Alignment</th><th>Total Score</th><th>Ethical Compliance</th></tr>"
        for rank, entry in enumerate(agent3_data["complete_rankings"], start=1):
            new_leaderboard_html += f"<tr><td>{rank}</td><td>{entry['model_name']}</td><td>{entry['accuracy']}</td><td>{entry['ethical_alignment']}</td><td>{entry['total_score']}</td><td>{entry['ethical_compliance']}</td></tr>"
        new_leaderboard_html += "</table>"
    except KeyError as e:
        print(f"Error: Missing key in agent data: {e}")
        return
    except Exception as e:
        print(f"Error creating leaderboard HTML: {e}")
        return

    try:
        # Read the existing index.html file and replace the leaderboard
        with open("index.html", "r") as f:
            html_content = f.read()
        
        # Replace old leaderboard table
        if "<table" in html_content and "</table>" in html_content:
            start_idx = html_content.index("<table")
            end_idx = html_content.index("</table>") + len("</table>")
            html_content = html_content[:start_idx] + new_leaderboard_html + html_content[end_idx:]
        else:
            html_content += new_leaderboard_html
        
        # Add a timestamp for the update
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"<p>Last updated: {timestamp}</p>"
        
        # Write back to the index.html file
        with open("index.html", "w") as f:
            f.write(html_content)
    except FileNotFoundError:
        print("Error: 'index.html' file not found.")
        return
    except Exception as e:
        print(f"Error updating 'index.html': {e}")
        return

    try:
        # Commit and push changes
        subprocess.run(["git", "add", "."], check=True, cwd=os.path.join(os.path.dirname(__file__), "github-pages"))
        subprocess.run(["git", "commit", "-m", "Updated evaluation results"], check=True, cwd=os.path.join(os.path.dirname(__file__), "github-pages"))
        subprocess.run(["git", "push", "origin", BRANCH], check=True, cwd=os.path.join(os.path.dirname(__file__), "github-pages"))
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return


if __name__ == "__main__":
    update_github_pages()
