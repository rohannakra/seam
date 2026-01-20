from google import genai
import os
import sys
import json
import numpy as np

client = genai.Client(api_key="")   

def llm_output(hitter, pitcher):

    with open("LLM/seam_data.json", "r") as f:
        seam_data = json.load(f)
    
    z_values = [point.get('z', 0) for point in seam_data]
    threshold = np.percentile(z_values, 90)  # Top 10% of density
    
    high_density = [point for point in seam_data if point.get('z', 0) > threshold]

    prompt = f"""
    
    You are a baseball analyst, given SEAM data, a predictive batted-ball distrubution of a hypothetical matchup between {hitter} and {pitcher} with coordinates (x, y) on a baseball field with density (z).
    This data contains the highest density regions, otherwise known as the regions of the field where the ball is most likely to land for the hypothetical matchup between {hitter} and {pitcher}

    SEAM data: {high_density}

    Requirements:
        - only include 2-3 bullet points and no other text above or below it
        - 100 - 150 characters for each bullet point
        - include a new line between each bullet point
        - make sure to include 2-3 DETAILED statistics
        - These statistics provide evidence for the SEAM data. For example, you should say "[statistic] aligns with SEAM data..." not "SEAM data aligns with [statistic]" 
        - Include a newline after each bullet point

    Here are some ideas for each bullet point:
        - hitter's pull rate
        - hitter's exit velocity in specific quadrants of the strike zone
        - pitcher's pitch placement
        - pitcher's pitch selection
        - cite matchup(s) between {hitter} and {pitcher} (outcome of matchup(s), batter/pitcher approach throughout at-bat, etc.)

    For example, if given hitter="Pete Crow-Armstrong", pitcher="Paul Skenes", the output may look like:

        * Crow-Armstrong was 7th in Pull % in 2025, confirming SEAM's highest density region being on Crow Armstrong's pull-side

        * In 12 matchups, Skenes commonly induces weak contact to the right side of the field aligning with almost all SEAM data being near 1B/RF

    For example, if given hitter="Javier Baez", pitcher="Matthew Boyd", the output may look like

        * Baez had a chase % of 46.1% in 2025, resulting in weak contact, corresponding with SEAM's highest density region lying in the infield

        * In their matchups, Boyd attacked Baez with fastballs and changeups outside of the strikezone to create weak infield pop-ups
    
    Notice how each example includes DETAILED statistics and data from matchups between the pitcher and hitter
    Also notice that each example shows how the statistic/previous matchups correlate with the SEAM data
    Also remember the (x, y) coordinates represent locations on a baseball field NOT a strikezone

    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text

if __name__ == "__main__":
    print(llm_output(sys.argv[1], sys.argv[2]))

# # ---- TESTING ----
# llm_output("Javier Baez", "Clayton Kershaw")    # NOTE: use this just to test that API is responsive

# NOTE: seems like should be 60 characters per bullet (measured 58).
