import axios from "axios"

export const SERVER_URL = "http://127.0.0.1:8000"
export const CLIENT_URL = "http://127.0.0.1:3000"

export class Api {
    private static request = axios.create({
        baseURL:`${SERVER_URL}/marketai/`,
        withCredentials: false,
        headers: {
            "Content-Type": "application/json",
            /*"Access-Control-Allow-Origin": CLIENT_URL,
            "Access-Control-Allow-Methods":"POST",
            "Access-Control-Allow-Headers":"Origin, X-Requested-With, Content-Type, Accept, Authorization" */
        }
    });

    static async post <T>(url: string, body?: T) {
        const finalUrl = `${SERVER_URL}/${url}/`;
        const res = await this.request.post(finalUrl, body)
            .then(function (res){
                return res
            })
            .catch(function (err) {
                alert("Errore nella richiesta al server, riprovare più tardi.."+err)
            })
        return res
    }
}