int frame = 1;
PShape frameS;

void setup() {
  size(1280, 720);
  frameRate(12);
}

int scale = 40;
float radius = 2;

void draw() {
  background(255);
  stroke(0);
  translate(220, 0);
  //scale(scale);
  
  fill(0);
  String[] lines;
  lines = loadStrings("../output_" + frame);
  if (lines == null){
    delay(5000);
    exit();
  } else {
    for (int i = 0; i < lines.length; i++) {
      String[] pieces = split(lines[i], ",");
      frameS = createShape(ELLIPSE, Float.parseFloat(pieces[0])*scale,Float.parseFloat(pieces[1])*scale, radius, radius);
      shape(frameS, 0, 0);
    }
  }
  frame++;
  //Uncomment to save each frame to a jpg file
  //saveFrame("out-######.jpg");
  delay(600);
}