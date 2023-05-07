import React, { useEffect, useState } from "react";
// import './Home.css';
import "./Upload.css";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Spinner from "./Spinner";
// import Output from './Output';

const imageMimeType = /image\/(png|jpg|jpeg)/i;
var base64String = "";

function Upload() {
  const [isLoading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [fileDataURL, setFileDataURL] = useState(null);

  const changeHandler = (e) => {
    const file = e.target.files[0];
    if (!file.type.match(imageMimeType)) {
      alert("Image mime type is not valid");
      return;
    }
    setFile(file);
  };
  useEffect(() => {
    let fileReader,
      isCancel = false;
    if (file) {
      base64String = "";
      fileReader = new FileReader();
      fileReader.onload = (e) => {
        const { result } = e.target;
        base64String = fileReader.result
          .replace("data:", "")
          .replace(/^.+,/, "");
        if (result && !isCancel) {
          setFileDataURL(result);
        }
      };
      fileReader.readAsDataURL(file);
    }
    return () => {
      isCancel = true;
      if (fileReader && fileReader.readyState === 1) {
        fileReader.abort();
      }
    };
  }, [file]);

  const data = {
    data: base64String,
  };

  const jsonifiedData = JSON.stringify(data);

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      window.res = await axios.post(
        "http://localhost:5000/predictGradcam",
        jsonifiedData
      );
      if (window.res.data.success) {
        setLoading(false);
        navigate("/output");
      }
    } catch (err) {
      console.log(err.message);
    }
  };

  return (
    <div className="uploading">
      {isLoading ? (
        <Spinner />
      ) : (
        <div className="uploading-child">
          <form onSubmit={handleSubmit}>
            <p>
              <label htmlFor="image"> Upload X-ray: </label>
              <input
                type="file"
                id="image"
                accept=".png, .jpg, .jpeg"
                onChange={changeHandler}
              />
            </p>
            <br />
            {fileDataURL ? (
              <p className="img-preview-wrapper">
                {<img src={fileDataURL} alt="preview" width="300px" />}
              </p>
            ) : null}{" "}
            <br />
            <input id="submit" type="submit" />
          </form>
          <br />
        </div>
      )}
    </div>
  );
}
export default Upload;
