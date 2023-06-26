// import NavBar1 from "./NavBar";
import "./Home.css";
// import Upload from "./Upload";

function Home() {
  return (
    <div className="wrapper">
      <div className="static-txt">Diagnose</div>
      <ul className="dynamic-txt">
        <li>Pneumonia</li>
        <li>Online</li>
        <li>it yourself</li>
      </ul>
      <div className="button">
        <a href="./upload">
          <button>Upload your photo</button>
        </a>
      </div>
    </div>
  );
}

export default Home;
