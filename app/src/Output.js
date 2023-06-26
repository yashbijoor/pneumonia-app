import NavBar1 from "./NavBar";
import "./Output.css";
import originalImage from "./Images/image.jpg";
import gradcamImage from "./Images/xray_cam.jpg";
import limeImage from "./Images/result-lime.png";

function Output() {
  return (
    <div className="outputs">
      <NavBar1 />
      <br />
      <h3>Here are your results:</h3>
      <h3>{window.res.data["success"]}</h3>
      <div className="output-wrapper">
        <div>
          <p>Original Image:</p>
          <img
            className="output-img"
            src={originalImage}
            alt="Uploaded xray"
          ></img>
        </div>
        <div>
          <p>Grad-CAM:</p>
          <img
            className="output-img"
            src={gradcamImage}
            alt="GradCAM heatmap"
          ></img>
        </div>
        <div>
          <p>LIME:</p>
          <img
            className="output-img"
            src={limeImage}
            alt="LIME generated"
          ></img>
        </div>
      </div>
      <div>
        <a href="/Upload">
          <button>Upload another X-ray</button>
        </a>
      </div>
    </div>
  );
}

export default Output;
