import os
import uvicorn
import logging
from fastapi import FastAPI, status
from fastapi import File, UploadFile
from fastapi import Response, Request
from fastapi import HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from pydantic import BaseModel
import TF_OD_Opencv as tfod

objectDetector = tfod.ObjectDetector("CenterNet HourGlass104 Keypoints 512x512")
logger = logging.getLogger(__name__)
app = FastAPI()

class Settings(BaseModel):
    authjwt_secret_key: str = os.getenv('JWT_SECRET')

@AuthJWT.load_config
def get_config():
    return Settings()


@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request: Request, exc: AuthJWTException):
    return JSONResponse(
        status_code=exc.status_code,
        content={'detail': exc.message}
    )

@app.post("/predict/image", status_code=status.HTTP_200_OK)
async def predict_api(
    response: Response,
    request: Request,
    img_file: UploadFile = File(None),
    Authorize: AuthJWT = Depends()
):
    if not (img_file):
        raise HTTPException(status_code=400, detail="Must specify one image file")
    elif img_file:
        if img_file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file must have the ´.jpeg´ or ´.png´ file extension.",
            )
        img_to_segment = await img_file.read()

    try:
        np_image = objectDetector.ImgToNpArray(img_to_segment)
        inference = objectDetector.Inference(np_image)
        img_path = objectDetector.DetectionImage(inference, np_image)
    except Exception as e:
        logging.exception("Object Detection Exception ", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Exception while executing Image Inference :({e})"
        )
    return FileResponse(img_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
