from dashboard.app import app as dash_app

app = dash_app.server


if __name__ == '__main__':
    dash_app.run(debug=False, port=8050)
