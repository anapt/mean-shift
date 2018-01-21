int frame = 1;
PShape frameS;

void setup() {
  size(720, 720);
  frameRate(12);
}

int scale = 1;
float radius = 2;
float maxX = 17.124000;
float minX = 3.402000;
float maxY = 14.996000;
float minY = 3.178000;

void draw() {
  background(255);
  stroke(0);
  //scale(scale);
  
  fill(0);
  System.out.println("frame = " + frame);
  String[] lines;
  lines = loadStrings("../output_" + frame);
  if (lines == null){
    delay(5000);
    exit();
  } else {
    for (int i = 0; i < lines.length; i++) {
      String[] pieces = split(lines[i], ",");
      float mapedX = map(Float.parseFloat(pieces[0]), minX, maxX, 0, 720);
      float mapedY = map(Float.parseFloat(pieces[1]), minY, maxY, 0, 720);
      frameS = createShape(ELLIPSE, mapedX*scale, mapedY*scale, radius, radius);
      shape(frameS, 0, 0);
    }
  }
  frame++;
  
  //Uncomment to save each frame to a jpg file
  //saveFrame("out-######.jpg");
  delay(600);
}