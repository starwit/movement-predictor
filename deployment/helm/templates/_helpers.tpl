{{/*
Expand the name of the chart.
*/}}
{{- define "ande.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ande.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ande.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ande.labels" -}}
helm.sh/chart: {{ include "ande.chart" . }}
{{ include "ande.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ande.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ande.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ande.serviceAccountName" -}}
{{- include "ande.fullname" . }}
{{- end }}

{{/*
Derive redis service name
*/}}
{{- define "ande.redisServiceName" -}}
{{- printf "%s-%s" .Release.Name "redis-master" -}}
{{- end }}

{{/*
Derive redis metrics service name
*/}}
{{- define "ande.redisMetricsServiceName" -}}
{{- printf "%s-%s" .Release.Name "redis-metrics" -}}
{{- end }}

{{/*
Hard-code Redis service port (for now)
*/}}
{{- define "ande.redisServicePort" -}}
"6379"
{{- end }}

{{/*
Derive redis metrics service name
*/}}
{{- define "ande.nodeExporterServiceName" -}}
{{- printf "%s-%s" .Release.Name "nodeexporter" -}}
{{- end }}

{{/*
Derive prometheus service name
*/}}
{{- define "ande.prometheusServiceName" -}}
{{- printf "%s-%s" .Release.Name "prometheus-server" -}}
{{- end }}
