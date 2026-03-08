# Football Injury Risk Labeling Guide

## What Are We Doing?

We are teaching an AI to recognize when a football player might be at risk of injury based on how they move. We need humans to watch short video clips and label them: "this movement looks safe" or "this movement looks risky."

---

## Step 1: Collect Videos

### What to record
- Short clips of football movements (10-60 seconds each)
- Good examples to include:
  - **Running / sprinting**
  - **Stopping quickly (deceleration)**
  - **Changing direction (cutting)**
  - **Jumping and landing**
  - **Kicking**
  - **Tackling / being tackled**

### How to save
1. Create a folder: `data/football/raw/`
2. Name files clearly: `player_001_run.mp4`, `player_002_cut.mp4`
3. Keep clips short (30-60 seconds max)

---

## Step 2: Label Each Clip

### Two Questions Per Clip

**Question 1: Is the player at risk?**
| Label | Meaning |
|-------|---------|
| `0` = Healthy | Good technique, safe movement |
| `1` = At Risk | Poor technique, injury warning sign |

**Question 2: Which body part? (optional)**
| Label | Body Part |
|-------|-----------|
| `left_knee` | Left knee |
| `right_knee` | Right knee |
| `left_ankle` | Left ankle |
| `right_ankle` | Right ankle |
| `left_hamstring` | Left hamstring |
| `right_hamstring` | Right hamstring |
| `spine` | Back / spine |
| `other` | Something else |

### Example Labels

| Video File | Risk (0/1) | Body Part | Notes |
|------------|------------|-----------|-------|
| player_001_run.mp4 | 0 | - | Good form |
| player_002_cut.mp4 | 1 | left_knee | Knee caving inward |
| player_003_jump.mp4 | 0 | - | Clean landing |
| player_004_land.mp4 | 1 | right_ankle | Heavy landing |

---

## Step 3: Use Label Studio (Easiest)

### What is Label Studio?
A free tool that lets you label data through a web browser. No coding needed (One Time)

1. Go to.

### Setup: https://labelstudio.io/
2. Download and install Label Studio
3. Open Label Studio in your browser (usually http://localhost:9090)
4. Create an account

### Create a Project

1. Click **Create Project**
2. Name it: "Football Injury Risk"
3. Click **Data Import** → Upload all your video files
4. Click **Labeling Config** → Copy and paste this:

```xml
<View>
  <Header value="Is this player at risk of injury?"/>
  <Choices name="risk" toName="video">
    <Choice value="0 - Healthy" />
    <Choice value="1 - At Risk" />
  </Choices>
  <Header value="Which body part is at risk?"/>
  <Choices name="region" toName="video">
    <Choice value="none" />
    <Choice value="left_knee" />
    <Choice value="right_knee" />
    <Choice value="left_ankle" />
    <Choice value="right_ankle" />
    <Choice value="left_hamstring" />
    <Choice value="right_hamstring" />
    <Choice value="spine" />
    <Choice value="other" />
  </Choices>
  <TextArea name="notes" toName="video"
            placeholder="Add any notes about this clip..."
            rows="3" />
</View>
```

5. Click **Save**

### Start Labeling

1. Click **Label** to open a video
2. Watch the video (play button)
3. Select:
   - **Risk**: 0 (Healthy) or 1 (At Risk)
   - **Region**: Body part at risk (or "none")
4. Click **Submit** → Next video
5. Repeat!

### Export Labels

When done:
1. Click **Export**
2. Choose **CSV**
3. Save the file as `football_labels.csv`

---

## Step 4: Convert to Training Format

Once labeling is done, we (the technical team) will:
1. Convert your CSV labels to the format the AI needs
2. Extract skeleton data from videos automatically
3. Combine into training files

You'll just give us: `football_labels.csv` + all video files

---

## Summary

| Step | What You Do | Output |
|------|-------------|--------|
| 1 | Record short football clips | Videos in a folder |
| 2 | Watch each clip, label risk (0/1) + body part | Label Studio project |
| 3 | Export as CSV | `football_labels.csv` |
| 4 | Give us the videos + CSV | We handle the rest |

---

## Need Help?

- **Video too long?** → Trim to 30-60 seconds
- **Unclear if at risk?** → Ask another person to label same clip
- **Technical issues?** → Contact: [your email]

---

**Thank you for helping!**
